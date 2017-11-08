export class ConvGPUBenchmark {

    constructor(params) {
        this.params = params
    }

    run(size) {
        const gpgpu = new GPGPUContext();
        const texManager = new TextureManager(gpgpu);
        initializeGPU(gpgpu, texManager);

        const inDepth = this.params.inDepth;
        const inShape = [size, size, inDepth];
        const outDepth = this.params.outDepth;
        const filterSize = this.params.filterSize;
        const stride = this.params.stride;
        const hasBias = true;
        const convInfo = conv_util.computeConvInfo(
            inShape, filterSize, filterSize, outDepth, stride, stride, 'same');
        const program = new Conv2DProgram(convInfo, hasBias);
        const outputShape = program.outputShape;
        const out = Array3D.zeros(outputShape);
        const x = Array3D.randUniform(inShape, -1, 1);
        const wShape =
            conv_util.computeWeightsShape4D(1, outDepth, filterSize, filterSize);
        const W = Array4D.randUniform(wShape, -1, 1);
        const b = Array1D.randUniform([outDepth], -1, 1);
        const inputs = [x, W, b];
        const binary = gpgpu_math.compileProgram(gpgpu, program, inputs, out);

        const benchmark = () => {
            gpgpu_math.runProgram(binary, inputs, out);
        };

        const cleanup = () => {
            x.dispose();
            W.dispose();
            b.dispose();
            out.dispose();
            texManager.dispose();
            gpgpu.deleteProgram(binary.webGLProgram);
            gpgpu.dispose();
        };

        // Warmup.
        await gpgpu.runQuery(benchmark);

        let totalTime;

        if (ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE')) {
            totalTime = await gpgpu.runQuery(benchmark);
        } else {
            const start = performance.now();

            benchmark();
            out.dataSync();

            totalTime = performance.now() - start;
        }
        cleanup();
        return totalTime;
    }
}


const convParams = {
    inDepth: 8,
    outDepth: 3,
    filterSize: 7,
    stride: 1
};

bmrun = new ConvGPUBenchmark(convParams)

t = bmrun.run()

console.log('benchmark result', t)