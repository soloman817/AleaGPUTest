[<AutoOpen>]
module Test.FS.Common

open System
open Alea.CUDA
open NUnit.Framework

let startTest (testName:string) =
    printfn ""
    printfn "=====> %s <=====" testName

let assertArrayClose (expected:'T[]) (actual:'T[]) (error:float) =
    (expected, actual) ||> Array.iter2 (fun expected actual -> Assert.That(actual, Is.EqualTo(expected).Within(error)))
