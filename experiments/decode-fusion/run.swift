import Foundation
import Metal

// Dependency-free local microbenchmark. This is not an end-to-end Uzu benchmark.
let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let library = try device.makeLibrary(URL: URL(fileURLWithPath: CommandLine.arguments[1]))
let projection = try device.makeComputePipelineState(function: library.makeFunction(name: "baseline_projection")!)
let gate = try device.makeComputePipelineState(function: library.makeFunction(name: "baseline_gate")!)
let fusedName = ProcessInfo.processInfo.environment["PARALLEL_PAIRS"] == "1" ? "paired_parallel_projection_gate" : "paired_projection_gate"
let fused = try device.makeComputePipelineState(function: library.makeFunction(name: fusedName)!)
func buffer(_ size: Int) -> MTLBuffer { device.makeBuffer(length: size, options: .storageModeShared)! }
func bf16(_ value: Float) -> UInt16 {
    let bits = value.bitPattern
    return UInt16(truncatingIfNeeded: (bits &+ 0x7fff &+ ((bits >> 16) & 1)) >> 16)
}
func float(_ value: UInt16) -> Float { Float(bitPattern: UInt32(value) << 16) }
var rng: UInt64 = 0x5eed
func random() -> UInt32 {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17
    return UInt32(truncatingIfNeeded: rng)
}
let k = 2560, h = 9216
precondition(k % 32 == 0 && h % 128 == 0)
let weights = buffer(2*h*k/2), scales = buffer(2*h*k/32*2), zeros = buffer(2*h*k/64)
let input = buffer(k*2), intermediate = buffer(2*h*2), baselineOutput = buffer(h*2), fusedOutput = buffer(h*2)
let outputFactors = buffer(2*h*4), inputFactors = buffer(h*4)
for i in 0..<weights.length/4 { weights.contents().assumingMemoryBound(to: UInt32.self)[i] = random() }
for i in 0..<scales.length/2 { scales.contents().assumingMemoryBound(to: UInt16.self)[i] = bf16(Float(random()%100+1)/10000) }
for i in 0..<zeros.length { zeros.contents().assumingMemoryBound(to: UInt8.self)[i] = UInt8(truncatingIfNeeded: random()) }
for i in 0..<outputFactors.length/4 { outputFactors.contents().assumingMemoryBound(to: Int32.self)[i] = random()%2 == 0 ? -1 : 1 }
for i in 0..<inputFactors.length/4 { inputFactors.contents().assumingMemoryBound(to: Int32.self)[i] = random()%2 == 0 ? -1 : 1 }
if let directory = ProcessInfo.processInfo.environment["MODEL_TENSORS"] {
    for (name,destination) in [("weights",weights),("scales",scales),("zeros",zeros),("output_factors",outputFactors),("input_factors",inputFactors)] {
        let data = try Data(contentsOf:URL(fileURLWithPath:directory).appendingPathComponent(name+".bin"))
        precondition(data.count == destination.length,"tensor size mismatch")
        data.withUnsafeBytes { bytes in destination.contents().copyMemory(from:bytes.baseAddress!,byteCount:data.count) }
    }
}
func setInput(_ magnitude: Float) {
    for i in 0..<k { input.contents().assumingMemoryBound(to: UInt16.self)[i] = bf16((Float(random()%20001)/10000-1)*magnitude) }
}
func dispatch(_ enc: MTLComputeCommandEncoder, _ pipeline: MTLComputePipelineState, _ src: MTLBuffer, _ dst: MTLBuffer, _ groups: Int, _ threads: Int) {
    enc.setComputePipelineState(pipeline)
    for (i,b) in [weights,scales,zeros,src,dst,outputFactors,inputFactors].enumerated() { enc.setBuffer(b,offset:0,index:i) }
    var kk = UInt32(k), hh = UInt32(h)
    enc.setBytes(&kk,length:4,index:7); enc.setBytes(&hh,length:4,index:8)
    enc.dispatchThreadgroups(MTLSize(width:groups,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:threads,height:1,depth:1))
}
func execute(_ optimized: Bool, _ iterations: Int) throws -> (Double,Double) {
    let start = DispatchTime.now().uptimeNanoseconds
    let command = queue.makeCommandBuffer()!
    command.label = optimized ? "paired projection + gate" : "projection then gate"
    let enc = command.makeComputeCommandEncoder()!
    for _ in 0..<iterations {
        if optimized { dispatch(enc,fused,input,fusedOutput,h/32,256) }
        else {
            dispatch(enc,projection,input,intermediate,2*h/32,256)
            dispatch(enc,gate,intermediate,baselineOutput,h/128,128)
        }
    }
    enc.endEncoding(); command.commit(); command.waitUntilCompleted()
    if let error = command.error { throw error }
    return ((command.gpuEndTime-command.gpuStartTime)*1e6/Double(iterations),Double(DispatchTime.now().uptimeNanoseconds-start)/1000/Double(iterations))
}
var checks: [[String:Any]] = []
for magnitude: Float in [0,0.01,1,8] {
    setInput(magnitude)
    _ = try execute(false,1); _ = try execute(true,1)
    let a = baselineOutput.contents().assumingMemoryBound(to: UInt16.self), b = fusedOutput.contents().assumingMemoryBound(to: UInt16.self)
    var mismatch = 0, maxError: Float = 0
    for i in 0..<h {
        if a[i] != b[i] { mismatch += 1 }
        let av = float(a[i]), bv = float(b[i])
        precondition(av.isFinite && bv.isFinite)
        maxError = max(maxError,abs(av-bv))
    }
    checks.append(["input_magnitude":magnitude,"bit_mismatches":mismatch,"max_abs_error":maxError])
    precondition(mismatch == 0,"BF16 bit mismatch")
}
setInput(1)
for _ in 0..<10 { _ = try execute(false,1); _ = try execute(true,1) }
var samples: [[String:Any]] = []
for iterations in [1,32] {
    for repetition in 0..<15 {
        for optimized in (repetition%2 == 0 ? [false,true] : [true,false]) {
            let (gpu,wall) = try execute(optimized,iterations)
            samples.append(["optimized":optimized,"repetition":repetition,"iterations":iterations,"gpu_us":gpu,"wall_us":wall,"thermal_state":ProcessInfo.processInfo.thermalState.rawValue])
        }
    }
}
if CommandLine.arguments.count > 2 {
    let descriptor = MTLCaptureDescriptor()
    descriptor.captureObject = device
    descriptor.destination = .gpuTraceDocument
    descriptor.outputURL = URL(fileURLWithPath: CommandLine.arguments[2])
    try MTLCaptureManager.shared().startCapture(with: descriptor)
    _ = try execute(false,1); _ = try execute(true,1)
    MTLCaptureManager.shared().stopCapture()
}
for i in 0..<h {
    precondition(baselineOutput.contents().assumingMemoryBound(to:UInt16.self)[i] == fusedOutput.contents().assumingMemoryBound(to:UInt16.self)[i], "repeated dispatch mismatch")
}
let result: [String:Any] = ["device":device.name,"fused_kernel":fusedName,"model_tensors":ProcessInfo.processInfo.environment["MODEL_TENSORS"] ?? "synthetic","k":k,"h":h,"warmup_pairs":10,"checks":checks,"samples":samples,"note":"W4/group32 unsigned weights (see model_tensors); synthetic input activations; BF16 RHT/SiLU/RHT; reused weights; GPU command-buffer time; no model or decode trace; excludes compile."]
print(String(data:try JSONSerialization.data(withJSONObject:result,options:[.prettyPrinted,.sortedKeys]),encoding:.utf8)!)
