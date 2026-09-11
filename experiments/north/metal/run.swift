import Foundation
import Metal
let device=MTLCreateSystemDefaultDevice()!
let queue=device.makeCommandQueue()!
let library=try device.makeLibrary(URL:URL(fileURLWithPath:CommandLine.arguments[1]))
let pipeline=try device.makeComputePipelineState(function:library.makeFunction(name:"branch")!)
func bf16(_ f:Float)->UInt16 { let b=f.bitPattern;return UInt16(truncatingIfNeeded:(b &+ 0x7fff &+ ((b>>16)&1))>>16) }
let D=2048,H=768,E=8,A=4096
let names=["x","residual","ax","up","gate","down","aw","scores","hidden","partial","attention","output","head","visits"]
let sizes=[D*2,D*2,A*2,E*H*D*2,E*H*D*2,E*D*H*2,D*A*2,E*4,E*H*2,E*24*D*4,D*2,D*2,4,256*4]
let buffers=sizes.map { device.makeBuffer(length:$0,options:.storageModeShared)! }
var seed:UInt64=12345
func random()->Float { seed ^= seed<<13;seed ^= seed>>7;seed ^= seed<<17;return Float(seed%2001)/1000-1 }
let tensorPath=ProcessInfo.processInfo.environment["NORTH_TENSORS"]
for index in 0..<8 {
 if let path=tensorPath {
  let data=try Data(contentsOf:URL(fileURLWithPath:path).appendingPathComponent(names[index]+".bin"))
  precondition(data.count==sizes[index],"tensor shape")
  data.withUnsafeBytes { buffers[index].contents().copyMemory(from:$0.baseAddress!,byteCount:$0.count) }
 } else if index==7 {
  for i in 0..<E { buffers[index].contents().assumingMemoryBound(to:Float.self)[i]=Float(i+1)/36 }
 } else {
  let p=buffers[index].contents().assumingMemoryBound(to:UInt16.self)
  for i in 0..<sizes[index]/2 { p[i]=bf16(random()*(index>=3 ? 0.02:1)) }
 }
}
let choices:[(String,UInt32)]=[("phased",256),("local_fusion",256),("mixed_grid",256),("mixed_interleaved",256),
 ("static",20),("dynamic",20),("static",64),("dynamic",64),("static",128),("static_interleaved",128),("dynamic",128),("static",256),("static_interleaved",256),("dynamic",256)]
func execute(_ name:String,_ workers:UInt32)throws->(Double,Double) {
 let start=DispatchTime.now().uptimeNanoseconds
 let cb=queue.makeCommandBuffer()!;cb.label="north-branch/"+name
 let blit=cb.makeBlitCommandEncoder()!
 for i in [12,13] {blit.fill(buffer:buffers[i],range:0..<sizes[i],value:0)}
 blit.endEncoding()
 let phases:[(UInt32,Int)]
 switch name {
 case "phased":phases=[(0,192),(1,192),(2,64),(3,8)]
 case "local_fusion":phases=[(4,192),(2,64),(3,8)]
 case "mixed_grid","mixed_interleaved":phases=[(4,256),(3,8)]
 case "static","static_interleaved":phases=[(5,Int(workers)),(3,8)]
 default:phases=[(6,Int(workers)),(3,8)]
 }
 for (phase,groups) in phases {
  let enc=cb.makeComputeCommandEncoder()!;enc.setComputePipelineState(pipeline)
  for (i,b) in buffers.enumerated(){enc.setBuffer(b,offset:0,index:i)}
  var cfg:[UInt32]=[phase,workers,name.contains("interleaved") ? 1:0,0]
  enc.setBytes(&cfg,length:16,index:14)
  enc.dispatchThreadgroups(MTLSize(width:groups,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:256,height:1,depth:1))
  enc.endEncoding()
 }
 cb.commit();cb.waitUntilCompleted()
 if let error=cb.error {throw error}
 return ((cb.gpuEndTime-cb.gpuStartTime)*1e6,Double(DispatchTime.now().uptimeNanoseconds-start)/1000)
}
func snapshot()->[Data] { [8,9,10,11].map {Data(bytes:buffers[$0].contents(),count:sizes[$0])} }
func check(_ expected:[Data],_ name:String) {
 precondition(snapshot()==expected,"intermediate/output mismatch")
 if name != "phased" && name != "local_fusion" {
  for i in 0..<256 {precondition(buffers[13].contents().assumingMemoryBound(to:UInt32.self)[i]==1,"visit count")}
 }
}
_ = try execute("phased",256)
let expected=snapshot()
var checks=0
for (name,workers) in choices {
 for _ in 0..<3 {
  // Poison intermediate/output buffers before correctness submissions.
  for i in [8,9,10,11] {memset(buffers[i].contents(),0x7f,sizes[i])}
  _ = try execute(name,workers);check(expected,name);checks+=1
 }
}
if let out=ProcessInfo.processInfo.environment["NORTH_OUTPUT"] {
 try expected[3].write(to:URL(fileURLWithPath:out))
}
var samples:[[String:Any]]=[]
if !CommandLine.arguments.contains("--checks-only") {
 for (name,w) in choices {for _ in 0..<5 {_ = try execute(name,w)}}
 for repetition in 0..<15 {
  var order=Array(choices.indices)
  let shift=repetition%order.count;order=Array(order[shift...]+order[..<shift])
  if repetition%2==1{order.reverse()}
  for i in order {
   let (name,w)=choices[i];let (gpu,wall)=try execute(name,w);check(expected,name)
   samples.append(["variant":name,"workers":w,"repetition":repetition,"gpu_us":gpu,"wall_us":wall,"thermal_state":ProcessInfo.processInfo.thermalState.rawValue])
  }
 }
}
let result:[String:Any]=["device":device.name,"tensors":tensorPath ?? "synthetic","checks":checks,"samples":samples,
 "note":"North dimensions. Selected 8 expert branches plus attention output projection. BF16 materialized weights. Local up/gate/down-partial chains; global residual join in second dispatch. No full attention, routing or quantized weight-streaming here. All variants share same split-K math. GPU/host time includes counter reset."]
print(String(data:try JSONSerialization.data(withJSONObject:result,options:[.prettyPrinted,.sortedKeys]),encoding:.utf8)!)
