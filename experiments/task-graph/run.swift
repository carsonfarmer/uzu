import Foundation
import Metal

struct Task {
    var id: UInt32, op: UInt32, a: UInt32, b: UInt32, dependencies: UInt32
    var unused0: UInt32 = 0, unused1: UInt32 = 0, unused2: UInt32 = 0
}
struct Config { var jobs: UInt32, width: UInt32, workers: UInt32, phase: UInt32 }
precondition(MemoryLayout<Task>.stride == 32 && MemoryLayout<Config>.stride == 16)
let tasks: [Task] = [
    Task(id:0,op:0,a:0,b:0,dependencies:0),
    Task(id:1,op:1,a:0,b:0,dependencies:1),
    Task(id:2,op:2,a:0,b:0,dependencies:1),
    Task(id:3,op:3,a:1,b:2,dependencies:6),
    Task(id:4,op:4,a:3,b:0,dependencies:8),
    Task(id:5,op:3,a:4,b:2,dependencies:20),
]
func validate(_ descriptors: [Task]) {
    precondition(descriptors.count == 6 && Set(descriptors.map { $0.id }) == Set(UInt32(0)..<UInt32(6)))
    for t in descriptors {
        precondition(t.op <= 4 && t.a < 6 && t.b < 6 && t.dependencies < 64)
        precondition(t.dependencies & (1 << t.id) == 0)
        if t.id != 0 { precondition(t.dependencies & (1 << t.a) != 0) }
        if t.op == 3 { precondition(t.dependencies & (1 << t.b) != 0) }
    }
    var done: UInt32 = 0
    for _ in 0..<6 {
        for t in descriptors where done & t.dependencies == t.dependencies { done |= 1 << t.id }
    }
    precondition(done == 63, "cyclic graph")
}
validate(tasks)
let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let library = try device.makeLibrary(URL:URL(fileURLWithPath:CommandLine.arguments[1]))
let variants = ["phased","fixed_fusion","static_tasks","dynamic_tasks"]
var pipelines: [String:MTLComputePipelineState] = [:]
for name in variants { pipelines[name] = try device.makeComputePipelineState(function:library.makeFunction(name:name)!) }
let width = 64, capacity = 4096
func allocate(_ bytes: Int) -> MTLBuffer { device.makeBuffer(length:bytes,options:.storageModeShared)! }
let input = allocate(capacity*width*4), scratch = allocate(capacity*6*width*4)
let counts = allocate(capacity*6*4), completed = allocate(capacity*4), head = allocate(4)
let descriptors = allocate(6*32)
func prepare(_ jobs: Int, _ step: Int, _ order: [Int]) {
    precondition(jobs > 0 && jobs <= capacity)
    let ordered = order.map { tasks[$0] }
    validate(ordered)
    ordered.withUnsafeBytes { descriptors.contents().copyMemory(from:$0.baseAddress!,byteCount:$0.count) }
    for i in 0..<jobs*width {
        input.contents().assumingMemoryBound(to:Float.self)[i] = Float((i*7+step*3)%129-64)/8
    }
}
func check(_ jobs: Int) {
    let x = input.contents().assumingMemoryBound(to:Float.self)
    let values = scratch.contents().assumingMemoryBound(to:Float.self)
    let visits = counts.contents().assumingMemoryBound(to:UInt32.self)
    let done = completed.contents().assumingMemoryBound(to:UInt32.self)
    for job in 0..<jobs {
        precondition(done[job] == 63, "unfinished graph")
        for task in 0..<6 { precondition(visits[job*6+task] == 1, "duplicate or missing task") }
        for lane in 0..<width {
            let a = x[job*width+lane]*2
            let b = a+1, c = a*a, d = b+c, e = d*0.5, f = e+c
            for (slot,expected) in [a,b,c,d,e,f].enumerated() {
                precondition(values[(job*6+slot)*width+lane].bitPattern == expected.bitPattern, "wrong intermediate")
            }
        }
    }
}
func execute(_ name: String, _ jobs: Int, _ workers: Int) throws -> [String:Double] {
    precondition(workers > 0 && workers <= 256)
    let start = DispatchTime.now().uptimeNanoseconds
    let command = queue.makeCommandBuffer()!
    command.label = "task-graph/"+name
    // GPU-side reset is included in both GPU command-buffer and host wall timings.
    let reset = command.makeBlitCommandEncoder()!
    for b in [counts,completed,head] { reset.fill(buffer:b,range:0..<b.length,value:0) }
    // Poison all intermediate values to catch consumption before production.
    reset.fill(buffer:scratch,range:0..<scratch.length,value:0x7f)
    reset.endEncoding()
    let phases = name == "phased" ? 6 : 1
    for phase in 0..<phases {
        let enc = command.makeComputeCommandEncoder()!
        enc.label = name + "/phase-" + String(phase)
        enc.setComputePipelineState(pipelines[name]!)
        for (index,buffer) in [input,scratch,counts,completed,head,descriptors].enumerated() {
            enc.setBuffer(buffer,offset:0,index:index)
        }
        var cfg = Config(jobs:UInt32(jobs),width:UInt32(width),workers:UInt32(workers),phase:UInt32(phase))
        enc.setBytes(&cfg,length:MemoryLayout<Config>.stride,index:6)
        let groups = (name == "phased" || name == "fixed_fusion") ? jobs : workers
        enc.dispatchThreadgroups(MTLSize(width:groups,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:width,height:1,depth:1))
        enc.endEncoding()
    }
    command.commit()
    command.waitUntilCompleted()
    if let error = command.error { throw error }
    return ["gpu_us":(command.gpuEndTime-command.gpuStartTime)*1e6,
            "wall_us":Double(DispatchTime.now().uptimeNanoseconds-start)/1000]
}
let orderings = [Array(0..<6),[5,4,3,2,1,0],[3,1,5,2,4,0]]
var checks = 0
for jobs in [1,19,257] {
    for workers in [1,7,20,64,256] {
        for (orderIndex,order) in orderings.enumerated() {
            for name in variants {
                if orderIndex != 0 && (name == "phased" || name == "fixed_fusion") { continue }
                prepare(jobs,orderIndex+workers,order)
                _ = try execute(name,jobs,workers)
                check(jobs)
                checks += 1
            }
        }
    }
}
// Vary input and descriptor order across repeated steps, resetting on GPU.
for step in 0..<24 {
    let name = step%2 == 0 ? "dynamic_tasks" : "static_tasks"
    prepare(257,step,orderings[step%3])
    _ = try execute(name,257,[1,7,20,64,256][step%5])
    check(257)
    checks += 1
}
var samples: [[String:Any]] = []
if !CommandLine.arguments.contains("--checks-only") {
    let choices: [(String,Int)] = [("phased",20),("fixed_fusion",20)] +
        [4,8,16,20,32,64,128,256].flatMap { [("static_tasks",$0),("dynamic_tasks",$0)] }
    prepare(capacity,1,orderings[0])
    for (name,workers) in choices {
        for _ in 0..<5 { _ = try execute(name,capacity,workers) }
    }
    for repetition in 0..<15 {
        // Rotate all variants, reversing every other repetition to reduce order bias.
        var order = Array(choices.indices)
        let shift = repetition % order.count
        order = Array(order[shift...] + order[..<shift])
        if repetition%2 == 1 { order.reverse() }
        for index in order {
            let (name,workers) = choices[index]
            let times = try execute(name,capacity,workers)
            check(capacity)
            samples.append(["variant":name,"workers":workers,"repetition":repetition,
                "gpu_us":times["gpu_us"]!,"wall_us":times["wall_us"]!,
                "thermal_state":ProcessInfo.processInfo.thermalState.rawValue])
        }
    }
}
let result: [String:Any] = [
    "device":device.name,"jobs":capacity,"width":width,"tasks_per_job":6,
    "correctness_runs":checks,"every_intermediate_bitwise_exact":true,
    "task_visits_exactly_once":true,"descriptor_orders":orderings,
    "samples":samples,
    "scope":"Synthetic six-op diamond DAG. Dynamic queue claims whole independent graphs; dependencies stay inside one threadgroup. No cross-threadgroup dependency scheduler, model weights, prefetch, or full-model decode.",
    "timing":"GPU-side counter reset and scratch poison included. Host descriptor construction/input upload and post-run verification excluded. Shared device scratch for all variants."
]
print(String(data:try JSONSerialization.data(withJSONObject:result,options:[.prettyPrinted,.sortedKeys]),encoding:.utf8)!)
