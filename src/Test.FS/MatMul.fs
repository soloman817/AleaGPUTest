module Test.FS.MatMul

open System
open Alea.CUDA
open NUnit.Framework

type JITModule(target, blockSize) =
    inherit GPUModule(target)

    [<Kernel;ReflectedDefinition>]
    member this.GPUKernel (C:deviceptr<float32>) (A:deviceptr<float32>) (B:deviceptr<float32>) (wA:int) (wB:int) =
        // Block index
        let bx = blockIdx.x
        let by = blockIdx.y

        // Thread index
        let tx = threadIdx.x
        let ty = threadIdx.y

        // Index of the first sub-matrix of A processed by the block
        let aBegin = wA * blockSize * by

        // Index of the last sub-matrix of A processed by the block
        let aEnd = aBegin + wA - 1

        // Step size used to iterate through the sub-matrices of A
        let aStep = blockSize

        // Index of the first sub-matrix of B processed by the block
        let bBegin = blockSize * bx

        // Step size used to iterate through the sub-matrices of B
        let bStep = blockSize * wB

        // Csub is used to store the element of the block sub-matrix
        // that is computed by the thread
        let mutable Csub = 0.0f

        // Loop over all the sub-matrices of A and B
        // required to compute the block sub-matrix
        let mutable a = aBegin
        let mutable b = bBegin
        while a <= aEnd do
            
            // Declaration of the shared memory array As used to
            // store the sub-matrix of A
            let As = __shared__.Array2D(blockSize, blockSize)

            // Declaration of the shared memory array Bs used to
            // store the sub-matrix of B
            let Bs = __shared__.Array2D(blockSize, blockSize)

            // Load the matrices from device memory
            // to shared memory; each thread loads
            // one element of each matrix
            As.[ty, tx] <- A.[a + wA * ty + tx]
            Bs.[ty, tx] <- B.[b + wB * ty + tx]

            // Synchronize to make sure the matrices are loaded
            __syncthreads()

            // Multiply the two matrices together;
            // each thread computes one element
            // of the block sub-matrix
            for k = 0 to blockSize - 1 do
                Csub <- Csub + As.[ty, k] * Bs.[k, tx]

            // Synchronize to make sure that the preceding
            // computation is done before loading two new
            // sub-matrices of A and B in the next iteration
            __syncthreads()

            a <- a + aStep
            b <- b + bStep

        // Write the block sub-matrix to device memory;
        // each thread writes one element
        let c = wB * blockSize * by + blockSize * bx
        C.[c + wB * ty + tx] <- Csub

    member this.GPUCalc (A:float32[]) (B:float32[]) (wA:int) (wB:int) =
        let worker = this.GPUWorker

        let wC = wB
        let hC = A.Length / wA

        use A = worker.Malloc(A)
        use B = worker.Malloc(B)
        use C = worker.Malloc<float32>(wC * hC)

        let threads = dim3(blockSize, blockSize)
        let grid = dim3(wB / threads.x, hC / threads.y)
        let lp = LaunchParam(grid, threads)
        this.GPULaunch <@ this.GPUKernel @> lp C.Ptr A.Ptr B.Ptr wA wB
        C.Gather()
            
    member this.CPUKernel (C:float32[]) (A:float32[]) (B:float32[]) (wA:int) (wB:int) =
        let hA = A.Length / wA
        for i = 0 to hA - 1 do
            for j = 0 to wB - 1 do
                let mutable sum = 0.0f
                for k = 0 to wA - 1 do
                    let a = A.[i * wA + k]
                    let b = B.[k * wB + j]
                    sum <- sum + a * b
                C.[i * wB + j] <- sum

    member this.CPUCalc (A:float32[]) (B:float32[]) (wA:int) (wB:int) =
        let hA = A.Length / wA
        let C = Array.zeroCreate<float32> (hA * wB)
        this.CPUKernel C A B wA wB
        C           
            
    member this.Test(dimA:int*int, dimB:int*int) =
        let worker = this.GPUWorker

        printfn "[Matrix Multiply Using CUDA] - Starting..."
        printfn "GPU Device %d: %A with compute capability %d.%d"
                worker.Device.ID worker.Device.Name
                worker.Device.Arch.Major worker.Device.Arch.Minor
        printfn ""

        let wA, hA = dimA
        let wB, hB = dimB

        let sizeA = wA * hA
        let sizeB = wB * hB

        printfn "MatrixA(%d,%d), MatrixB(%d,%d)" wA hA wB hB

        let A = Array.init sizeA (fun _ -> 1.0f)
        let B = Array.init sizeB (fun _ -> 0.01f)

        let hOutput = this.CPUCalc A B wA wB
        printfn "Computing result using CUDA Kernel..."
        let dOutput = this.GPUCalc A B wA wB
        printfn "done"

        // do performance test
        let wC = wB
        let hC = A.Length / wA

        use A = worker.Malloc(A)
        use B = worker.Malloc(B)
        use C = worker.Malloc<float32>(wC * hC)

        let threads = dim3(blockSize, blockSize)
        let grid = dim3(wB / threads.x, hC / threads.y)
        let lp = LaunchParam(grid, threads)

        worker.Synchronize()
        let nIter = 300
        use start = worker.CreateEvent()
        use stop = worker.CreateEvent()
        let launch = this.GPULaunch <@ this.GPUKernel @>
        start.Record()
        for i = 1 to nIter do
            launch lp C.Ptr A.Ptr B.Ptr wA wB
        stop.Record()
        stop.Synchronize()
        let msecTotal = Event.ElapsedMilliseconds(start, stop)

        // Compute and print the performance
        let msecPerMatrixMul = msecTotal / float(nIter)
        let flopsPerMatrixMul = 2.0 * float(wA) * float(hA) * float(wB)
        let gigaFlops = (flopsPerMatrixMul * 1.0e-9) / (msecPerMatrixMul / 1000.0)
        printfn "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block"
                gigaFlops msecPerMatrixMul flopsPerMatrixMul (threads.x * threads.y)

        printf "Checking computed result for correctness: "
        assertArrayClose hOutput dOutput 1e-5
        printfn "Result = PASS"

    member this.Test() =
        let dimA = 320, 320
        let dimB = 640, 320
        this.Test(dimA, dimB)

[<AOTCompile(AOTOnly = true)>]
type AOTModule(target) = inherit JITModule(target, 32)

let testJIT(device:Device) =
    startTest "MatMul.JIT"
    use worker = Worker.Create(device)
    use m = new JITModule(GPUModuleTarget.Worker(worker), 32)
    m.Test()

let testAOT(device:Device) =
    startTest "MatMul.AOT"
    use worker = Worker.Create(device)
    use m = new AOTModule(GPUModuleTarget.Worker(worker))
    m.Test()

