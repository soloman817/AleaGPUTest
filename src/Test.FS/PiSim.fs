module Test.FS.PiSim

open System
open Alea.CUDA
open Alea.CUDA.Unbound
open NUnit.Framework

// parameters of one calc task
type CalcParam =
    { RunId : int
      NumPoints : int
      NumStreamsPerSM : int
      GetRandom : int -> int -> Rng.IRandom<float> }

type JITModule(target) =
    inherit GPUModule(target)

    let reduceModule = new DeviceSumModuleI32(target)
    let rngModuleXorShift7 = new Rng.XorShift7.CUDA.DefaultUniformRandomModuleF64(target)
    let rngModuleMrg32k3a = new Rng.Mrg32k3a.CUDA.DefaultUniformRandomModuleF64(target)

    override this.Dispose(disposing) =
        if disposing then
            reduceModule.Dispose()
            rngModuleXorShift7.Dispose()
            rngModuleMrg32k3a.Dispose()
        base.Dispose(disposing)

    [<Kernel;ReflectedDefinition>]
    member this.KernelCountInside (pointsX:deviceptr<float>) (pointsY:deviceptr<float>) (numPoints:int) (numPointsInside:deviceptr<int>) =
        let start = blockIdx.x * blockDim.x + threadIdx.x
        let stride = gridDim.x * blockDim.x
        let mutable i = start
        while i < numPoints do
            let x = pointsX.[i]
            let y = pointsY.[i]
            numPointsInside.[i] <- if sqrt (x*x + y*y) <= 1.0 then 1 else 0
            i <- i + stride

    member this.CalcPi (param:CalcParam) =
        let worker = this.GPUWorker
        // we switch to the gpu worker thread to do the job
        worker.Eval <| fun _ ->
            let numPoints = param.NumPoints
            let numStreamsPerSM = param.NumStreamsPerSM
            let numSMs = worker.Device.Attributes.MULTIPROCESSOR_COUNT
            let numStreams = numStreamsPerSM * numSMs
            let numDimensions = 2

            let random = param.GetRandom numStreams numDimensions
            use reduce = reduceModule.Create(numPoints)
            use points = random.AllocCUDAStreamBuffer(numPoints)
            use numPointsInside = worker.Malloc<int>(numPoints)
            let pointsX = points.Ptr
            let pointsY = points.Ptr + numPoints
            let lp = LaunchParam(numSMs * 8, 256)

            printfn "Run #.%d : Random(%s) Streams(%d) Points(%d)" param.RunId (random.GetType().Namespace) numStreams numPoints

            // run on all random streams
            [| 0..numStreams-1 |]
            |> Array.map (fun streamId ->
                random.Fill(streamId, numPoints, points)
                this.GPULaunch <@ this.KernelCountInside @> lp pointsX pointsY numPoints numPointsInside.Ptr
                let numPointsInside = reduce.Reduce(numPointsInside.Ptr, numPoints)
                4.0 * (float numPointsInside) / (float numPoints) )
            |> Array.average

    // a function to create a list of calcPI task parameter. We randomly select the seed and rng.
    member this.CreateParams (numPoints:int) (numStreamsPerSM:int) (numRuns:int) : CalcParam[] =
        let rng = Random()
        Array.init numRuns (fun runId ->
            let seed = rng.Next() |> uint32
            let getRandomXorshift7 numStreams numDimensions = rngModuleXorShift7.Create(numStreams, numDimensions, seed) :> Rng.IRandom<float>
            let getRandomMrg32k3a  numStreams numDimensions = rngModuleMrg32k3a.Create(numStreams, numDimensions, seed) :> Rng.IRandom<float>
            let getRandom =
                match rng.Next(2) with
                | 0 -> getRandomXorshift7
                | _ -> getRandomMrg32k3a
            { RunId = runId; NumPoints = numPoints; NumStreamsPerSM = numStreamsPerSM; GetRandom = getRandom } )

    member this.Simulate numPoints numStreamsPerSM numRuns =
        this.CreateParams numPoints numStreamsPerSM numRuns
        |> Array.map this.CalcPi
        |> Array.average

    member this.Test() =
        let million = 1000000
        let numPoints = 10 * million
        let numStreamsPerSM = 10
        let numRuns = 20
        let pi = this.Simulate numPoints numStreamsPerSM numRuns
        printfn "PI = %A" pi

[<AOTCompile(AOTOnly = true)>]
type AOTModule(target) = inherit JITModule(target)

let testJIT(device:Device) =
    startTest "PiSim.JIT"
    use worker = Worker.Create(device)
    use m = new JITModule(GPUModuleTarget.Worker(worker))
    m.Test()

let testAOT(device:Device) =
    startTest "PiSim.AOT"
    use worker = Worker.Create(device)
    use m = new AOTModule(GPUModuleTarget.Worker(worker))
    m.Test()

