var dl = deeplearn;
var Graph = dl.Graph;
var Tensor = dl.Tensor;
var Scalar = dl.Scalar;
var NDArrayMathGPU = dl.NDArrayMathGPU;
var Session = dl.Session;
var track = dl.track;
var keep = dl.keep;
var InCPUMemoryShuffledInputProviderBuilder = dl.InCPUMemoryShuffledInputProviderBuilder;
var SGDOptimizer = dl.SGDOptimizer;
var CostReduction = dl.CostReduction;
var Array1D = dl.Array1D;

math = new NDArrayMathGPU();

/**
 * This implementation of computing the complementary color came from an
 * answer by Edd https://stackoverflow.com/a/37657940
 */
function computeComplementaryColor(rgbColor) {
    let r = rgbColor[0];
    let g = rgbColor[1];
    let b = rgbColor[2];

    // Convert RGB to HSL
    // Adapted from answer by 0x000f http://stackoverflow.com/a/34946092/4939630
    r /= 255.0;
    g /= 255.0;
    b /= 255.0;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    let h = (max + min) / 2.0;
    let s = h;
    const l = h;

    if (max === min) {
        h = s = 0; // achromatic
    } else {
        const d = max - min;
        s = (l > 0.5 ? d / (2.0 - max - min) : d / (max + min));

        if (max === r && g >= b) {
            h = 1.0472 * (g - b) / d;
        } else if (max === r && g < b) {
            h = 1.0472 * (g - b) / d + 6.2832;
        } else if (max === g) {
            h = 1.0472 * (b - r) / d + 2.0944;
        } else if (max === b) {
            h = 1.0472 * (r - g) / d + 4.1888;
        }
    }

    h = h / 6.2832 * 360.0 + 0;

    // Shift hue to opposite side of wheel and convert to [0-1] value
    h += 180;
    if (h > 360) {
        h -= 360;
    }
    h /= 360;

    // Convert h s and l values into r g and b values
    // Adapted from answer by Mohsen http://stackoverflow.com/a/9493060/4939630
    if (s === 0) {
        r = g = b = l; // achromatic
    } else {
        const hue2rgb = (p, q, t) => {
            if (t < 0) t += 1;
            if (t > 1) t -= 1;
            if (t < 1 / 6) return p + (q - p) * 6 * t;
            if (t < 1 / 2) return q;
            if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
            return p;
        };

        const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        const p = 2 * l - q;

        r = hue2rgb(p, q, h + 1 / 3);
        g = hue2rgb(p, q, h);
        b = hue2rgb(p, q, h - 1 / 3);
    }

    return [r, g, b].map(v => Math.round(v * 255));
}


function generateRandomChannelValue() {
    return Math.floor(Math.random() * 256);
}

function normalizeColor(rgbColor) {
    return rgbColor.map(v => v / 255);
}

function denormalizeColor(normalizedRgbColor) {
    return normalizedRgbColor.map(v => v * 255);
}

const rawInputs = new Array(1e5);
for (let i = 0; i < 1e5; i++) {
    rawInputs[i] = [
        generateRandomChannelValue(), generateRandomChannelValue(),
        generateRandomChannelValue()
    ];
}


const graph = new Graph();

// This tensor contains the input. In this case, it is a scalar.
inputTensor = graph.placeholder('input RGB value', [3]);

// This tensor contains the target.
targetTensor = graph.placeholder('output RGB value', [3]);

const inputArray =
    rawInputs.map(c => Array1D.new(normalizeColor(c)));
const targetArray = rawInputs.map(
    c => Array1D.new(
        normalizeColor(computeComplementaryColor(c))));


const shuffledInputProviderBuilder = new InCPUMemoryShuffledInputProviderBuilder([inputArray, targetArray]);
const [inputProvider, targetProvider] = shuffledInputProviderBuilder.getInputProviders();

feedEntries = [{
        tensor: inputTensor,
        data: inputProvider
    },
    {
        tensor: targetTensor,
        data: targetProvider
    }
];


function createFullyConnectedLayer(
    graph, inputLayer, layerIndex,
    sizeOfThisLayer, includeRelu = true, includeBias = true) {
    return graph.layers.dense(
        'fully_connected_' + layerIndex, inputLayer, sizeOfThisLayer,
        includeRelu ? (x) => graph.relu(x) : undefined, includeBias);
}

// Create 3 fully connected layers, each with half the number of nodes of
// the previous layer. The first one has 16 nodes.
let fullyConnectedLayer =
    createFullyConnectedLayer(graph, inputTensor, 0, 64);

// Create fully connected layer 1, which has 8 nodes.
fullyConnectedLayer =
    createFullyConnectedLayer(graph, fullyConnectedLayer, 1, 32);

// Create fully connected layer 2, which has 4 nodes.
fullyConnectedLayer =
    createFullyConnectedLayer(graph, fullyConnectedLayer, 2, 16);


predictionTensor =
    createFullyConnectedLayer(graph, fullyConnectedLayer, 3, 3);

costTensor =
    graph.meanSquaredCost(targetTensor, predictionTensor);



//////////////////////
///Train and Predict//
//////////////////////


session = new Session(graph, math);

var batchSize = 100;
var initialLearningRate = 0.02
optimizer = new SGDOptimizer(initialLearningRate);


function train1Batch(shouldFetchCost) {
    // Every 42 steps, lower the learning rate by 15%.
    const learningRate =
        initialLearningRate * Math.pow(0.95, Math.floor(step / 82));
    optimizer.setLearningRate(learningRate);

    // Train 1 batch.
    let costValue = -1;
    math.scope(() => {
        const cost = session.train(
            costTensor, feedEntries, batchSize, optimizer,
            shouldFetchCost ? CostReduction.MEAN : CostReduction.NONE);

        if (!shouldFetchCost) {
            // We only train. We do not compute the cost.
            return;
        }

        // Compute the cost (by calling get), which requires transferring data
        // from the GPU.
        costValue = cost.get();
    });
    return costValue;
}



function predict(rgbColor) {
    let complementColor = [];
    math.scope((keep, track) => {
        const mapping = [{
            tensor: inputTensor,
            data: Array1D.new(normalizeColor(rgbColor)),
        }];
        const evalOutput = session.eval(predictionTensor, mapping);
        const values = evalOutput.getValues();
        const colors = denormalizeColor(Array.prototype.slice.call(values));

        // Make sure the values are within range.
        complementColor = colors.map(
            v => Math.round(Math.max(Math.min(v, 255), 0)));
    });
    return complementColor;
}


function populateContainerWithColor(
    container, r, g, b) {
    const originalColorString = 'rgb(' + [r, g, b].join(',') + ')';
    container.textContent = originalColorString;

    const colorBox = document.createElement('div');
    colorBox.classList.add('color-box');
    colorBox.style.background = originalColorString;
    container.appendChild(colorBox);
}

var UI_initialized = false

function initializeUi() {
    const colorRows = document.querySelectorAll('tr[data-original-color]');
    for (let i = 0; i < colorRows.length; i++) {
        const rowElement = colorRows[i];
        const tds = rowElement.querySelectorAll('td');
        const originalColor =
            (rowElement.getAttribute('data-original-color'))
            .split(',')
            .map(v => parseInt(v, 10));

        // Visualize the original color.
        populateContainerWithColor(
            tds[0], originalColor[0], originalColor[1], originalColor[2]);

        // Visualize the complementary color.
        const complement =
            computeComplementaryColor(originalColor);
        populateContainerWithColor(
            tds[1], complement[0], complement[1], complement[2]);
    }


    var d = document.getElementById('egdiv');
    d.innerHTML = 'step = ' + step;

    UI_initialized = true
}

// On every frame, we train and then maybe update the UI.
let step = 0;


var plot_exist = false

var data_x = [];
var data_y = [];
var data_z = [];


function create_plot3d(init_x, init_y, init_z) {

    Plotly.d3.csv('https://raw.githubusercontent.com/plotly/datasets/master/3d-line1.csv', function (err, rows) {

        Plotly.plot('graph-color', [{
                type: 'scatter3d',
                mode: 'lines',
                x: init_x,
                y: init_y,
                z: init_z,
                opacity: 1,
                line: {
                    width: 6,
                    // color: c,
                    reversescale: false
                },
                displayModeBar: false
            }]
            // , {
            //     height: 640
            // }
        );

    });

}

function update_plot3d(new_x, new_y, new_z) {

    Plotly.d3.csv('https://raw.githubusercontent.com/plotly/datasets/master/3d-line1.csv', function (err, rows) {
        Plotly.animate('graph-color', {
            data: [{
                x: new_x,
                y: new_y,
                z: new_z
            }],
            traces: [0],
            layout: {}
        }, {
            transition: {
                duration: 500,
                easing: 'cubic-in-out'
            }
        })

    });

}


function train_per() {
    if (step > 4242) {
        // Stop training.
        return;
    }

    if (paused) return;

    // We only fetch the cost every 5 steps because doing so requires a transfer
    // of data from the GPU.

    cost = train1Batch(step % 10 === 0);

    var d = document.getElementById('egdiv');
    d.innerHTML = 'step = ' + step;

    if (step % 10 === 0) {

        // Print data to console so the user can inspect.
        console.log('step', step - 1, 'cost', cost);

        // Visualize the predicted complement.
        const colorRows = document.querySelectorAll('tr[data-original-color]');
        for (let i = 0; i < colorRows.length; i++) {
            const rowElement = colorRows[i];
            const tds = rowElement.querySelectorAll('td');
            const originalColor =
                (rowElement.getAttribute('data-original-color'))
                .split(',')
                .map(v => parseInt(v, 10));

            // Visualize the predicted color.
            const predictedColor = predict(originalColor);
            populateContainerWithColor(
                tds[2], predictedColor[0], predictedColor[1], predictedColor[2]);

            if (i === 0) {
                data_x.push(predictedColor[0])
                data_y.push(predictedColor[1])
                data_z.push(predictedColor[2])
                if (plot_exist === false) {
                    create_plot3d(data_x, data_y, data_z);
                    plot_exist = true
                } else {
                    update_plot3d(data_x, data_y, data_z);
                }
            }


        }
    }

    step++;
}



var toggle_pause = function () {
    paused = !paused;
    var btn = document.getElementById('buttontp');
    if (paused) {
        btn.value = 'Resume'
    } else {
        btn.value = 'Pause';
    }
}

var update_net_param_display = function () {
    initializeUi();
    // document.getElementById('lr_input').value = trainer.learning_rate;
    // document.getElementById('momentum_input').value = trainer.momentum;
    // document.getElementById('batch_size_input').value = trainer.batch_size;
    // document.getElementById('decay_input').value = trainer.l2_decay;
}


var paused = true;


function start() {
    if (UI_initialized) {
        console.log('starting!');
        setInterval(train_per, 5); // lets go!
    } else {
        update_net_param_display();
        console.log('waiting!');
        // load_data_batch(0);
        setTimeout(start, 1000); // run start again after 1second
    } // keep checking
}