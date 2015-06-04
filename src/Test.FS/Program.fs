module Test.FS.Program

open System
open Alea.CUDA
open NUnit.Framework

[<EntryPoint>]
let main argv = 
    if argv.Length = 2 || argv.Length = 3 then
        let deviceId = Int32.Parse argv.[0]
        let compileType = argv.[1]
        let testName = if argv.Length = 3 then argv.[2] else "all"
        let device = Device.DeviceDict.[deviceId]

        printfn " Device : %A" device
        printfn "  Cores : %A" device.Cores
        printfn "Default : %A" (deviceId = Device.Default.ID)
        printfn "Compile : %s" compileType
        printfn "   Test : %s" testName

        match compileType with
        | "jit" ->
            match testName with
            | "simple" -> Simple.testJIT(device); 0
            | "matmul" -> MatMul.testJIT(device); 0
            | "pisim" -> PiSim.testJIT(device); 0
            | "all" ->
                Simple.testJIT(device)
                MatMul.testJIT(device)
                PiSim.testJIT(device)
                0
            | _ -> printfn "Unknown test name %A." testName; -1

        | "aot" ->
            match testName with
            | "simple" -> Simple.testAOT(device); 0
            | "matmul" -> MatMul.testAOT(device); 0
            | "pisim" -> PiSim.testAOT(device); 0
            | "all" ->
                Simple.testAOT(device)
                MatMul.testAOT(device)
                PiSim.testAOT(device)
                0
            | _ -> printfn "Unknown test name %A." testName; -1

        | _ -> printfn "Unknown compile type %A." compileType; -1

    else 
        printfn "Test.FS [deviceid] [jit|aot] [testname]"
        -1
