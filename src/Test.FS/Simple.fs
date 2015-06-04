module Test.FS.Simple

open System
open Alea.CUDA
open NUnit.Framework

type JITModule(target) =
    inherit GPUModule(target)

    [<Kernel;ReflectedDefinition>]
    member this.Kernel (outputs:deviceptr<int>) (inputs1:deviceptr<int>) (inputs2:deviceptr<int>) (n:int) =
        let start = blockIdx.x * blockDim.x + threadIdx.x
        let stride = gridDim.x * blockDim.x
        let mutable i = start
        while i < n do
            outputs.[i] <- inputs1.[i] + inputs2.[i]
            i <- i + stride

    member this.Test() =
        let worker = this.GPUWorker
        let n = 100000
        let lp = LaunchParam(16, 256)
        let rng = Random()
        let inputs1 = Array.init n (fun _ -> rng.Next(-100, 100))
        let inputs2 = Array.init n (fun _ -> rng.Next(-80, 80))
        let expected = (inputs1, inputs2) ||> Array.map2 (+)
        use inputs1 = worker.Malloc(inputs1)
        use inputs2 = worker.Malloc(inputs2)
        use outputs = worker.Malloc<int>(n)
        this.GPULaunch <@ this.Kernel @> lp outputs.Ptr inputs1.Ptr inputs2.Ptr n
        let outputs = outputs.Gather()
        Assert.AreEqual(expected, outputs)
        printfn "Adding %d integers passed." n

[<AOTCompile(AOTOnly = true)>]
type AOTModule(target) = inherit JITModule(target)

let testJIT(device:Device) =
    startTest "Simple.JIT"
    use worker = Worker.Create(device)
    use m = new JITModule(GPUModuleTarget.Worker(worker))
    m.Test()

let testAOT(device:Device) =
    startTest "Simple.AOT"
    use worker = Worker.Create(device)
    use m = new AOTModule(GPUModuleTarget.Worker(worker))
    m.Test()

