var dl = deeplearn;
var NDArrayMathGPU = dl.NDArrayMathGPU;
var Array3D = dl.Array3D;
var Array4D = dl.Array4D;
var Array1D = dl.Array1D;
var ENV = dl.ENV;

function computeWeightsShape4D(
    inputDepth, outputDepth, filterHeight,
    filterWidth) {
    return [filterHeight, filterWidth, inputDepth, outputDepth];
}

class ConvGPUBenchmark {

    constructor(params) {
        this.params = params
    }

    run(size) {
        const math = new NDArrayMathGPU();
        const gpgpu = math.getGPGPUContext();

        const inDepth = this.params.inDepth;
        const inShape = [size, size, inDepth];
        const outDepth = this.params.outDepth;
        const filterSize = this.params.filterSize;
        const stride = this.params.stride;

        const x = Array3D.randUniform(inShape, -1, 1);
        const wShape = computeWeightsShape4D(
            inDepth, outDepth, filterSize, filterSize);
        const W = Array4D.randUniform(wShape, -1, 1);
        const b = Array1D.randUniform([outDepth], -1, 1);

        let out;
        const benchmark = () => {
            out = math.conv2d(x, W, b, stride, 'same');
        };

        const cleanup = () => {
            x.dispose();
            W.dispose();
            b.dispose();
            out.dispose();
        };

        // Warmup.
        gpgpu.runQuery(benchmark);
        out.dispose();

        let totalTime;

        if (ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE')) {
            totalTime = gpgpu.runQuery(benchmark);
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
    stride: 2
};


function buildRunNumbersRow(values) {
    const runNumberRowElement = document.createElement('div');
    runNumberRowElement.className = 'run-numbers-row math-benchmark';

    for (let i = 0; i < values.length; i++) {
        const runNumberCellElement = document.createElement('div');
        runNumberCellElement.className = 'run-numbers-cell math-benchmark';
        runNumberCellElement.innerText = values[i];
        runNumberRowElement.appendChild(runNumberCellElement);
    }
    return runNumberRowElement;
}



function create_chart(sizes) {

    for (let i = 0; i < sizes.length; i++) {
        const hue = Math.floor(360 * i / sizes.length);
        datasets.push({
            data: chartData,
            fill: false,
            label: name,
            borderColor: `hsl(${hue}, 100%, 40%)`,
            backgroundColor: `hsl(${hue}, 100%, 70%)`,
            pointRadius: 0,
            pointHitRadius: 5,
            borderWidth: 1,
            lineTension: 0
        });
    }

    const chart = new Chart(context, {
        type: 'line',
        data: {
            datasets
        },
        options: {
            animation: {
                duration: 0
            },
            responsive: false,
            scales: {
                xAxes: [{
                    type: 'linear',
                    position: 'bottom',
                    ticks: {
                        min: 0,
                        max: sizes.length,
                        stepSize: 10,
                        callback: (label) => {
                            return +label;
                        }
                        // tslint:disable-next-line:no-any
                    } // Note: the typings for this are incorrect, cast as any.
                }],
                yAxes: [{
                    ticks: {
                        callback: (label, index, labels) => {
                            return `${label}ms`;
                        }
                    },
                }]
            },
            tooltips: {
                mode: 'label'
            },
            title: {
                text: name
            }
        }
    });

    return chart
}

var chartData = []
var rowValues = []
var datasets = [];
var name = 'conv'

var runNumbersTable = document.querySelectorAll('.run-numbers-table')[0];
runNumbersTable.innerHTML = '';
runNumbersTable.style.display = 'none';

const canvas = document.querySelectorAll('.run-plot')[0];
// Avoid to growing size of rendered chart.
canvas.width = 400;
canvas.height = 300;
const context = canvas.getContext('2d')


function start() {

    bmrun = new ConvGPUBenchmark(convParams)

    const runPromises = [];

    function test_size(size) {
        t = bmrun.run(size)
        // t.then(data => console.log('conv on size', size, 'time', data, 'us'))
        runPromises.push(t);
    }

    var sizes = [];
    for (let size = 1; size < 2048; size = size * 2) {
        test_size(size)
        sizes.push(size)
    }

    Promise.all(runPromises).then(results => {
        for (let i = 0; i < results.length; i++) {

            let resultString;
            let logString;
            let time = 0;
            let success = true;
            let size = sizes[i]
            try {
                time = results[i];
                resultString = time.toFixed(3) + 'ms';
                logString = resultString;
            } catch (e) {
                success = false;
                resultString = 'Error';
                logString = e.message;
            }

            if (time >= 0) {
                if (success) {
                    chartData.push({
                        x: size,
                        y: time
                    });
                }
                rowValues.push(resultString);
            }
            console.log(`[${size}]: ${logString}`);
        }
        runNumbersTable.appendChild(buildRunNumbersRow(rowValues));

    });

    chart = create_chart(sizes)

}