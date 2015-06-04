open System
open Alea.CUDA
open NUnit.Framework

type JITModule(target) =
    inherit GPUModule(target)

    [<Kernel;ReflectedDefinition>]
    member this.TestKernel (outputs:deviceptr<int>) (inputs:deviceptr<int>) =
        let tid = threadIdx.x
        outputs.[tid] <- inputs.[tid] + 1

    member this.Test() =
        let worker = this.GPUWorker
        let rng = Random()
        let n = 128
        let lp = LaunchParam(1, n)
        let inputs = Array.init n (fun _ -> rng.Next(-100, 100))
        let expected = inputs |> Array.map ((+) 1)
        use inputs = worker.Malloc(inputs)
        use outputs = worker.Malloc<int>(n)
        this.GPULaunch <@ this.TestKernel @> lp outputs.Ptr inputs.Ptr
        let outputs = outputs.Gather()
        Assert.AreEqual(expected, outputs)

[<AOTCompile(AOTOnly = true)>]
type AOTModule(target) =
    inherit JITModule(target)

let testJIT(device:Device) =
    use worker = Worker.Create(device)
    use m = new JITModule(GPUModuleTarget.Worker(worker))
    m.Test()

let testAOT(device:Device) =
    use worker = Worker.Create(device)
    use m = new AOTModule(GPUModuleTarget.Worker(worker))
    m.Test()

[<EntryPoint>]
let main argv = 
    if argv.Length = 2 then
        let deviceId = Int32.Parse argv.[0]
        let testName = argv.[1]
        let device = Device.DeviceDict.[deviceId]
        printfn "Device : %A" device
        printfn "Cores  : %A" device.Cores
        printfn "Default: %A" (deviceId = Device.Default.ID)
        printfn "Test   : %s" testName
        match testName with
        | "jit" -> testJIT(device); 0
        | "aot" -> testAOT(device); 0
        | _ -> printfn "Unknown test name %A" testName; -1
    else 
        printfn "Test.FS [DeviceId] [TestName]"
        -1
