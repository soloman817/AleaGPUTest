module Test.FS.UnitTests

open System
open Alea.CUDA
open NUnit.Framework

[<Test>]
let SimpleJIT() =
    Simple.testJIT(Device.Default)

[<Test>]
let SimpleAOT() =
    Simple.testAOT(Device.Default)

[<Test>]
let MatMulJIT() =
    MatMul.testJIT(Device.Default)

[<Test>]
let MatMulAOT() =
    MatMul.testAOT(Device.Default)

[<Test>]
let PiSimJIT() =
    PiSim.testJIT(Device.Default)

[<Test>]
let PiSimAOT() =
    PiSim.testAOT(Device.Default)
