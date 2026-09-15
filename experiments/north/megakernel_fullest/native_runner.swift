// Command-buffer GPU duration for an exported primitive; never decode throughput.
import Foundation
import Metal

struct Input: Decodable { let path: String }
struct Variant: Decodable { let name: String; let library: String; let workers: Int }
struct Manifest: Decodable {
    let variants: [Variant]
    let inputs: [Input]
    let outputBytes: Int
    let repetitions: Int
    let output: String
}
let path=CommandLine.arguments[1]
let manifest=try JSONDecoder().decode(Manifest.self,from:Data(contentsOf:URL(fileURLWithPath:path)))
guard let device=MTLCreateSystemDefaultDevice(),let queue=device.makeCommandQueue() else {fatalError("No Metal device")}
let pipelines=try manifest.variants.map { variant -> MTLComputePipelineState in
    let lib=try device.makeLibrary(URL:URL(fileURLWithPath:variant.library))
    guard let fn=lib.makeFunction(name:"check") else {fatalError("Missing check kernel")}
    return try device.makeComputePipelineState(function:fn)
}
let buffers=try manifest.inputs.map { input -> MTLBuffer in
    let data=try Data(contentsOf:URL(fileURLWithPath:input.path))
    return data.withUnsafeBytes { device.makeBuffer(bytes:$0.baseAddress!,length:data.count,options:.storageModeShared)! }
}
let output=device.makeBuffer(length:manifest.outputBytes,options:.storageModeShared)!
func emit(_ row:[String:Any]) throws {
    let data=try JSONSerialization.data(withJSONObject:row,options:[.sortedKeys])
    print(String(data:data,encoding:.utf8)!)
    fflush(stdout)
}
for (i,pipeline) in pipelines.enumerated() {
    try emit(["kind":"native_provenance","variant":manifest.variants[i].name,"device":device.name,"maxThreads":pipeline.maxTotalThreadsPerThreadgroup,"threadWidth":pipeline.threadExecutionWidth,"staticThreadgroupBytes":pipeline.staticThreadgroupMemoryLength,"scope":"GPU command-buffer duration, includes workspace fill when dispatching. Exported primitive, not model decode."])
}
let count=pipelines.count+1
precondition(manifest.repetitions % (2*count)==0)
for rep in -3..<manifest.repetitions {
    let shift=max(0,rep)%count
    var order=Array(shift..<count)+Array(0..<shift)
    if rep>=0 && (rep/count)%2==1 {order.reverse()}
    for index in order {
        let fillOnly=index==pipelines.count
        let name=fillOnly ? "fill_only" : manifest.variants[index].name
        let start=ProcessInfo.processInfo.systemUptime
        let command=queue.makeCommandBuffer()!
        let blit=command.makeBlitCommandEncoder()!
        blit.fill(buffer:output,range:0..<manifest.outputBytes,value:0)
        blit.endEncoding()
        if !fillOnly {
            let compute=command.makeComputeCommandEncoder()!
            compute.setComputePipelineState(pipelines[index])
            for (index,buffer) in buffers.enumerated(){compute.setBuffer(buffer,offset:0,index:index)}
            compute.setBuffer(output,offset:0,index:buffers.count)
            compute.dispatchThreadgroups(MTLSize(width:manifest.variants[index].workers,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:256,height:1,depth:1))
            compute.endEncoding()
        }
        command.commit();command.waitUntilCompleted()
        if let error=command.error {throw error}
        let wall=ProcessInfo.processInfo.systemUptime-start
        let gpu=command.gpuEndTime-command.gpuStartTime
        try emit(["kind":"native_timing","repetition":rep,"warmup":rep<0,"variant":name,"fillOnly":fillOnly,"wallSeconds":wall,"gpuSeconds":gpu,"gpuTimestampsAvailable":command.gpuStartTime>0 && gpu>0])
        if !fillOnly && rep==manifest.repetitions-1 {
            try Data(bytes:output.contents(),count:manifest.outputBytes).write(to:URL(fileURLWithPath:manifest.output+"-"+name+".bin"))
        }
    }
}
