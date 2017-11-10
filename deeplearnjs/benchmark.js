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

var chartDataX = [];
var chartData = [];
var rowValues = [];


window.chartColors = {
    red: 'rgb(255, 99, 132)',
    orange: 'rgb(255, 159, 64)',
    yellow: 'rgb(255, 205, 86)',
    green: 'rgb(75, 192, 192)',
    blue: 'rgb(54, 162, 235)',
    purple: 'rgb(153, 102, 255)',
    grey: 'rgb(201, 203, 207)'
};

var config = {
    type: 'line',
    data: {
        labels: chartDataX,
        // labels: ["January", "February", "March", "April", "May", "June", "July"],
        datasets: [{
            label: "GPU--deeplearnjs",
            backgroundColor: window.chartColors.red,
            borderColor: window.chartColors.red,
            data: chartData,
            fill: false,
            pointRadius: 0,
            pointHitRadius: 5,
            borderWidth: 1,
            lineTension: 0,
        }]
    },
    options: {
        animation: {
            duration: 0
        },
        responsive: true,
        title: {
            display: true,
            text: 'Conv Benchmark on input size'
        },
        tooltips: {
            mode: 'index',
            intersect: false,
        },
        hover: {
            mode: 'nearest',
            intersect: true
        },
        scales: {
            xAxes: [{
                display: true,
                // type: 'logarithmic',
                position: 'bottom',
                scaleLabel: {
                    display: true,
                    labelString: 'Input Image width or height (pixel)'
                },
            }],
            yAxes: [{
                display: true,
                scaleLabel: {
                    display: true,
                    labelString: 'time elapsed'
                },
                ticks: {
                    min: 0,
                    callback: (label, index, labels) => {
                        let num = Number(label).toFixed(2);
                        return `${num} ms`;
                    }
                }
            }]
        }
    }
};

function insert_into_table(size, time, table) {

    var len = table.rows.length;
    var row = table.insertRow(len);

    // row.style.height = "10px";

    var row_col1 = row.insertCell(0);
    row_col1.innerHTML = size.toString();
    // Insert New Column for Row1 at index '1'.
    var row_col2 = row.insertCell(1);
    row_col2.innerHTML = time.toString();

    row.style.fontSize = "12px";

}

function create_chart() {

    const canvas = document.getElementById("run-plot")
    canvas.width = 400;
    canvas.height = 300;
    const context = canvas.getContext('2d');

    window.line_chart = new Chart(context, config);
};

function create_table(index = 0) {

    // var runNumbersTable_list = document.querySelectorAll('.run-numbers-table');
    // runNumbersTable = runNumbersTable_list[index]

    // Create table.
    var table = document.getElementById('divTable');

    // var table = document.createElement('table');
    // Insert New Row for table at index '0'.
    var row1 = table.insertRow(0);
    // Insert New Column for Row1 at index '0'.
    var row1col1 = row1.insertCell(0);
    row1col1.innerHTML = 'Size';
    // Insert New Column for Row1 at index '1'.
    var row1col2 = row1.insertCell(1);
    row1col2.innerHTML = 'Time(ms)';


    var els = table.getElementsByTagName("td");
    for (var i = 0; i < els.length; i++) {
        els[i].style.fontSize = "12px";
        els[i].style.fontWeight = "bold";
        els[i].style.color = "#000000"
    }

    // Append Table into div.
    // var div = document.getElementById('divTable');
    // div.appendChild(table);

    window.line_table = table

}



var btn = document.getElementById('buttontp');

var paused = true;

var toggle_pause = function () {
    paused = !paused;
    if (paused) {
        btn.value = 'Run Benchmark'
        btn.disabled = false
    } else {
        btn.value = 'Running...';
        btn.disabled = true
    }
}

function run_test() {

    const convParams = {
        inDepth: 8,
        outDepth: 3,
        filterSize: 7,
        stride: 2
    };

    bmrun = new ConvGPUBenchmark(convParams)

    const runPromises = [];

    function test_size(size) {
        t = bmrun.run(size)
        // t.then(data => console.log('conv on size', size, 'time', data, 'us'))
        runPromises.push(t);
    }

    var sizes = [];
    for (let size = 1; size < 4000; size = size * 2) {
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
                    if ($.inArray(size, chartDataX) === -1) {
                        chartDataX.push(size.toString());
                    }

                    insert_into_table(size, time, window.line_table);
                }
                rowValues.push(resultString);
            }
            console.log(`[${size}]: ${logString}`);

            config.data.datasets.data = chartData;
            config.data.labels = chartDataX;

            window.line_chart.update();

        }
        // runNumbersTable.appendChild(buildRunNumbersRow(rowValues));

    });

    toggle_pause();
}


function run() {

    if (paused === true) {
        setTimeout(function () {
            run();
        }, 0);
    } else {
        setTimeout(function () {
            run_test();
        }, 0);
    }

}


function start() {

    create_chart();

    create_table();

    run();
}