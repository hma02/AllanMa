/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */


var dl = deeplearn;
var AdadeltaOptimiz = dl.AdadeltaOptimiz;
var AdagradOptimize = dl.AdagradOptimize;
var AdamaxOptimizer = dl.AdamaxOptimizer;
var AdamOptimizer = dl.AdamOptimizer;
var Array1D = dl.Array1D;
var Array3D = dl.Array3D;
var DataStats = dl.DataStats;
var FeedEntry = dl.FeedEntry;
var Graph = dl.Graph;
var GraphRunner = dl.GraphRunner;
var GraphRunnerEven = dl.GraphRunnerEven;
var InCPUMemoryShuf = dl.InCPUMemoryShuf;
var InMemoryDataset = dl.InMemoryDataset;
var MetricReduction = dl.MetricReduction;
var MomentumOptimiz = dl.MomentumOptimiz;
var NDArray = dl.NDArray;
var NDArrayMath = dl.NDArrayMath;
var NDArrayMathCPU = dl.NDArrayMathCPU;
var NDArrayMathGPU = dl.NDArrayMathGPU;
var Optimizer = dl.Optimizer;
var RMSPropOptimize = dl.RMSPropOptimize;
var Scalar = dl.Scalar;
var Session = dl.Session;
var SGDOptimizer = dl.SGDOptimizer;
var MomentumOptimizer = dl.MomentumOptimizer;
var Tensor = dl.Tensor;
var util = dl.util;
var xhr_dataset = dl.xhr_dataset;
var XhrDataset = dl.XhrDataset;
var XhrDatasetConfi = dl.XhrDatasetConfi;

var InCPUMemoryShuffledInputProviderBuilder = dl.InCPUMemoryShuffledInputProviderBuilder;


function getDisplayShape(shape) {
    return `[${shape}]`;
}

const DATASETS_CONFIG_JSON = 'deeplearnjs/mnist/model-builder-datasets-config.json';

/** How often to evaluate the model against test data. */
const EVAL_INTERVAL_MS = 1500;
/** How often to compute the cost. Downloading the cost stalls the GPU. */
const COST_INTERVAL_MS = 500;
/** How many inference examples to show when evaluating accuracy. */
const INFERENCE_EXAMPLE_COUNT = 1;
const INFERENCE_IMAGE_SIZE_PX = 100;
/**
 * How often to show inference examples. This should be less often than
 * EVAL_INTERVAL_MS as we only show inference examples during an eval.
 */
const INFERENCE_EXAMPLE_INTERVAL_MS = 3000;

// Smoothing factor for the examples/s standalone text statistic.
const EXAMPLE_SEC_STAT_SMOOTHING_FACTOR = .7;

const TRAIN_TEST_RATIO = 5 / 6;

const IMAGE_DATA_INDEX = 0;
const LABEL_DATA_INDEX = 1;

var Normalization = {
    NORMALIZATION_NEGATIVE_ONE_TO_ONE: 0,
    NORMALIZATION_ZERO_TO_ONE: 1,
    NORMALIZATION_NONE: 2
}

// tslint:disable-next-line:variable-name
// export let ModelBuilderPolymer = PolymerElement({
//     is: 'model-builder',
//     properties: {
//         inputShapeDisplay: String,
//         isValid: Boolean,
//         inferencesPerSec: Number,
//         inferenceDuration: Number,
//         examplesTrained: Number,
//         examplesPerSec: Number,
//         totalTimeSec: String,
//         applicationState: Number,
//         modelInitialized: Boolean,
//         showTrainStats: Boolean,
//         datasetDownloaded: Boolean,
//         datasetNames: Array,
//         selectedDatasetName: String,
//         modelNames: Array,
//         selectedOptimizerName: String,
//         optimizerNames: Array,
//         learningRate: Number,
//         momentum: Number,
//         needMomentum: Boolean,
//         gamma: Number,
//         needGamma: Boolean,
//         beta1: Number,
//         needBeta1: Boolean,
//         beta2: Number,
//         needBeta2: Boolean,
//         batchSize: Number,
//         selectedModelName: String,
var selectedNormalizationOption = Normalization.NORMALIZATION_NEGATIVE_ONE_TO_ONE;

//         // Stats
//         showDatasetStats: Boolean,
//         statsInputMin: Number,
//         statsInputMax: Number,
//         statsInputShapeDisplay: String,
//         statsLabelShapeDisplay: String,
//         statsExampleCount: Number,
//     }
// });

var ApplicationState = {
    IDLE: 1,
    TRAINING: 2
}


var isValid = null;
// totalTimeSec: string;
// applicationState: ApplicationState;
var modelInitialized = false; //: boolean;
// showTrainStats: boolean;
// selectedNormalizationOption: number;

// // Datasets and models.
// graphRunner: GraphRunner;
// graph: Graph;
var graphRunner = null;
var session = null;
var optimizer; //: Optimizer;
var xTensor; //: Tensor;
var labelTensor; //: Tensor;
var costTensor; //: Tensor;
var accuracyTensor; //: Tensor;
var predictionTensor; //: Tensor;

// datasetDownloaded: boolean;
// datasetNames: string[];
var selectedDatasetName;
var modelNames;
// selectedModelName: string;
// optimizerNames: string[];
var selectedOptimizerName;
var loadedWeights;
var dataSets;
//: {
//     [datasetName: string]: InMemoryDataset
// };
var dataSet;
var xhrDatasetConfigs;
//: {
//     [datasetName: string]: XhrDatasetConfig
// };
// datasetStats: DataStats[];
var learningRate; //: number;
var momentum; //: number;
// needMomentum: boolean;
var gamma; //: number;
// needGamma: boolean;
var beta1; //: number;
// needBeta1: boolean;
var beta2; //: number;
// needBeta2: boolean;
var batchSize; //: number;

// // Stats.
// showDatasetStats: boolean;
// statsInputRange: string;
// statsInputShapeDisplay: string;
// statsLabelShapeDisplay: string;
// statsExampleCount: number;

// // Charts.
var costChart; //: Chart;
var accuracyChart; //: Chart;
var examplesPerSecChart; //: Chart;
var costChartData; //: ChartPoint[];
var accuracyChartData; //: ChartPoint[];
var examplesPerSecChartData; //: ChartPoint[];

// trainButton: HTMLButtonElement;

// // Visualizers.
var inputNDArrayVisualizers;
var outputNDArrayVisualizers; //: NDArrayLogitsVisualizer[];

var inputShape;
var labelShape;
// examplesPerSec: number;
// examplesTrained: number;
// inferencesPerSec: number;
// inferenceDuration: number;

// inputLayer: ModelLayer;
var hiddenLayers = [];

var layersContainer;

var math; //: NDArrayMath;
// // Keep one instance of each NDArrayMath so we don't create a user-initiated
// // number of NDArrayMathGPU's.
// mathGPU: NDArrayMathGPU;
// mathCPU: NDArrayMathCPU;

function createChart(canvasId, _label, _data, min = null, max = null) {
    const context = (document.getElementById(canvasId)).getContext('2d');
    return new Chart(context, {
        type: 'line',
        data: {
            datasets: [{
                _data,
                fill: false,
                _label,
                pointRadius: 0,
                borderColor: 'rgba(75,192,192,1)',
                borderWidth: 1,
                lineTension: 0,
                pointHitRadius: 8
            }]
        },
        options: {
            animation: {
                duration: 0
            },
            responsive: true,
            scales: {
                xAxes: [{
                    type: 'linear',
                    position: 'bottom'
                }],
                yAxes: [{
                    ticks: {
                        min: 0,
                    }
                }]
            }
        }
    });
}

function recreateCharts() {
    costChartData = [];
    if (costChart != null) {
        costChart.destroy();
    }
    costChart =
        createChart('cost-chart', 'Cost', costChartData, 0);

    if (accuracyChart != null) {
        accuracyChart.destroy();
    }
    accuracyChartData = [];
    accuracyChart = createChart(
        'accuracy-chart', 'Accuracy', accuracyChartData, 0, 100);

    if (examplesPerSecChart != null) {
        examplesPerSecChart.destroy();
    }
    examplesPerSecChartData = [];
    examplesPerSecChart = createChart(
        'examplespersec-chart', 'Examples/sec', examplesPerSecChartData,
        0);
}


function setupDatasetStats() {
    var datasetStats = dataSet.getStats();
    var statsExampleCount = datasetStats[IMAGE_DATA_INDEX].exampleCount;
    var statsInputRange =
        `[${datasetStats[IMAGE_DATA_INDEX].inputMin}, ` +
        `${datasetStats[IMAGE_DATA_INDEX].inputMax}]`;
    var statsInputShapeDisplay = getDisplayShape(
        datasetStats[IMAGE_DATA_INDEX].shape);
    var statsLabelShapeDisplay = getDisplayShape(
        datasetStats[LABEL_DATA_INDEX].shape);
    var showDatasetStats = true;
}

function getTestData() {
    const data = dataSet.getData();
    if (data == null) {
        return null;
    }
    const [images, labels] = dataSet.getData();

    const start = Math.floor(TRAIN_TEST_RATIO * images.length);

    return [images.slice(start), labels.slice(start)];
}

function getTrainingData() {
    const [images, labels] = dataSet.getData();

    const end = Math.floor(TRAIN_TEST_RATIO * images.length);

    return [images.slice(0, end), labels.slice(0, end)];
}


function createOptimizer() {
    switch (selectedOptimizerName) {
        case 'sgd':
            {
                return new SGDOptimizer(+learningRate);
            }
        case 'momentum':
            {
                return new MomentumOptimizer(+learningRate, +momentum);
            }
        case 'rmsprop':
            {
                return new RMSPropOptimizer(+learningRate, +gamma);
            }
        case 'adagrad':
            {
                return new AdagradOptimizer(+learningRate);
            }
        case 'adadelta':
            {
                return new AdadeltaOptimizer(+learningRate, +gamma);
            }
        case 'adam':
            {
                return new AdamOptimizer(+learningRate, +beta1, +beta2);
            }
        case 'adamax':
            {
                return new AdamaxOptimizer(+learningRate, +beta1, +beta2);
            }
        default:
            {
                throw new Error(`Unknown optimizer "${selectedOptimizerName}"`);
            }
    }
}



function startTraining() {
    const trainingData = getTrainingData();
    const testData = getTestData();

    // Recreate optimizer with the selected optimizer and hyperparameters.
    optimizer = createOptimizer();

    if (this.isValid && (trainingData != null) && (testData != null)) {
        recreateCharts();
        graphRunner.resetStatistics();

        const trainingShuffledInputProviderGenerator =
            new InCPUMemoryShuffledInputProviderBuilder(trainingData);
        const [trainInputProvider, trainLabelProvider] =
        trainingShuffledInputProviderGenerator.getInputProviders();

        const trainFeeds = [{
                tensor: this.xTensor,
                data: trainInputProvider
            },
            {
                tensor: this.labelTensor,
                data: trainLabelProvider
            }
        ];

        const accuracyShuffledInputProviderGenerator =
            new InCPUMemoryShuffledInputProviderBuilder(testData);
        const [accuracyInputProvider, accuracyLabelProvider] =
        accuracyShuffledInputProviderGenerator.getInputProviders();

        const accuracyFeeds = [{
                tensor: this.xTensor,
                data: accuracyInputProvider
            },
            {
                tensor: this.labelTensor,
                data: accuracyLabelProvider
            }
        ];

        graphRunner.train(
            costTensor, trainFeeds, batchSize, optimizer,
            undefined /** numBatches */ , accuracyTensor, accuracyFeeds,
            batchSize, MetricReduction.MEAN, EVAL_INTERVAL_MS,
            COST_INTERVAL_MS);

        showTrainStats = true;
        applicationState = ApplicationState.TRAINING;
    }
}

function startInference() {
    const testData = getTestData();
    if (testData == null) {
        // Dataset not ready yet.
        return;
    }
    if (isValid && (testData != null)) {
        const inferenceShuffledInputProviderGenerator =
            new InCPUMemoryShuffledInputProviderBuilder(testData);
        const [inferenceInputProvider, inferenceLabelProvider] =
        inferenceShuffledInputProviderGenerator.getInputProviders();

        const inferenceFeeds = [{
                tensor: xTensor,
                data: inferenceInputProvider
            },
            {
                tensor: labelTensor,
                data: inferenceLabelProvider
            }
        ];

        graphRunner.infer(
            predictionTensor, inferenceFeeds, INFERENCE_EXAMPLE_INTERVAL_MS,
            INFERENCE_EXAMPLE_COUNT);
    }
}

function createModel() {

    /*
    TO BE customized
    */

    if (session != null) {
        session.dispose();
    }

    modelInitialized = false;
    if (isValid === false) {
        return;
    }

    var graph = new Graph();
    const g = graph;
    xTensor = g.placeholder('input', inputShape);
    labelTensor = g.placeholder('label', labelShape);

    let network = xTensor;

    for (let i = 0; i < hiddenLayers.length; i++) {
        let weights = null;
        if (loadedWeights != null) {
            weights = loadedWeights[i];
        }
        network = hiddenLayers[i].addLayer(g, network, i, weights);
    }
    predictionTensor = network;
    costTensor =
        g.softmaxCrossEntropyCost(predictionTensor, labelTensor);
    accuracyTensor =
        g.argmaxEquals(predictionTensor, labelTensor);

    var loadedWeights = null;

    var session = new Session(g, math);
    graphRunner.setSession(session);

    // startInference();

    modelInitialized = true;
}



function updateSelectedDataset(datasetName) {
    if (dataSet != null) {
        dataSet.removeNormalization(IMAGE_DATA_INDEX);
    }

    graphRunner.stopTraining();
    graphRunner.stopInferring();

    if (dataSet != null) {
        dataSet.dispose();
    }

    selectedDatasetName = datasetName;
    var selectedModelName = '';
    dataSet = dataSets[datasetName];
    var datasetDownloaded = false;
    var showDatasetStats = false;

    dataSet.fetchData().then(() => {
        datasetDownloaded = true;
        applyNormalization(selectedNormalizationOption);
        setupDatasetStats();
        if (isValid) {
            createModel();
            // startInference();
        }
        // Get prebuilt models.
        populateModelDropdown();
    });

    inputShape = dataSet.getDataShape(IMAGE_DATA_INDEX);
    labelShape = dataSet.getDataShape(LABEL_DATA_INDEX);

    layersContainer = document.querySelector('#hidden-layers');

    var inputLayer = document.querySelector('#input-layer');
    var t = document.createTextNode("input-layer out:" + getDisplayShape(inputShape));
    inputLayer.appendChild(t)

    labelShapeDisplay = getDisplayShape(labelShape);
    const costLayer = document.querySelector('#cost-layer');
    var t = document.createTextNode("cost-layer in:" + labelShapeDisplay + " out:" + labelShapeDisplay);
    costLayer.appendChild(t)

    const outputLayer = document.querySelector('#output-layer');
    var t = document.createTextNode("output-layer out:" + labelShapeDisplay);
    outputLayer.appendChild(t)

    // Setup the inference example container.
    // TODO(nsthorat): Generalize 
    const inferenceContainer =
        document.querySelector('#inference-container');
    inferenceContainer.innerHTML = '';
    inputNDArrayVisualizers = [];
    outputNDArrayVisualizers = [];
    for (let i = 0; i < INFERENCE_EXAMPLE_COUNT; i++) {
        const inferenceExampleElement = document.createElement('div');
        inferenceExampleElement.className = 'inference-example';

        // Set up the input visualizer.

        const ndarrayImageVisualizer = new NDArrayImageVisualizer(inferenceExampleElement);
        ndarrayImageVisualizer.setShape(this.inputShape);
        ndarrayImageVisualizer.setSize(
            INFERENCE_IMAGE_SIZE_PX, INFERENCE_IMAGE_SIZE_PX);

        inputNDArrayVisualizers.push(ndarrayImageVisualizer);
        // inferenceExampleElement.appendChild(ndarrayImageVisualizer);

        // Set up the output ndarray visualizer.
        const ndarrayLogitsVisualizer = new NDArrayLogitsVisualizer(inferenceExampleElement)
        document.createElement('ndarray-logits-visualizer');
        ndarrayLogitsVisualizer.initialize(
            INFERENCE_IMAGE_SIZE_PX, INFERENCE_IMAGE_SIZE_PX);
        outputNDArrayVisualizers.push(ndarrayLogitsVisualizer);
        // inferenceExampleElement.appendChild(ndarrayLogitsVisualizer);

        inferenceContainer.appendChild(inferenceExampleElement);
    }
}



function populateDatasets() {
    dataSets = {};
    xhr_dataset.getXhrDatasetConfig(DATASETS_CONFIG_JSON)
        .then(_xhrDatasetConfigs => {
                for (const datasetName in _xhrDatasetConfigs) {
                    if (_xhrDatasetConfigs.hasOwnProperty(datasetName)) {
                        dataSets[datasetName] =
                            new XhrDataset(_xhrDatasetConfigs[datasetName]);
                    }
                }
                var datasetNames = Object.keys(dataSets);
                var selectedDatasetName = datasetNames[0];
                xhrDatasetConfigs = _xhrDatasetConfigs;
                updateSelectedDataset(datasetNames[0]);
            },
            error => {
                throw new Error(`Dataset config could not be loaded: ${error}`);
            });
}


function applyNormalization(selectedNormalizationOption) {
    switch (selectedNormalizationOption) {
        case Normalization.NORMALIZATION_NEGATIVE_ONE_TO_ONE:
            {
                dataSet.normalizeWithinBounds(IMAGE_DATA_INDEX, -1, 1);
                break;
            }
        case Normalization.NORMALIZATION_ZERO_TO_ONE:
            {
                dataSet.normalizeWithinBounds(IMAGE_DATA_INDEX, 0, 1);
                break;
            }
        case Normalization.NORMALIZATION_NONE:
            {
                dataSet.removeNormalization(IMAGE_DATA_INDEX);
                break;
            }
        default:
            {
                throw new Error('Normalization option must be 0, 1, or 2');
            }
    }
    setupDatasetStats();
}


function populateModelDropdown() {
    const _modelNames = ['Custom'];

    const modelConfigs =
        xhrDatasetConfigs[selectedDatasetName].modelConfigs;
    for (const modelName in modelConfigs) {
        if (modelConfigs.hasOwnProperty(modelName)) {
            _modelNames.push(modelName);
        }
    }
    modelNames = _modelNames;
    selectedModelName = _modelNames[_modelNames.length - 1];
    updateSelectedModel(selectedModelName);
}

function validateModel() {
    let valid = true;
    for (let i = 0; i < hiddenLayers.length; ++i) {
        valid = valid && hiddenLayers[i].isValid();
    }
    if (hiddenLayers.length > 0) {
        const lastLayer = hiddenLayers[hiddenLayers.length - 1];
        valid = valid &&
            util.arraysEqual(labelShape, lastLayer.getOutputShape());
    }
    isValid = valid && (hiddenLayers.length > 0);
}

function layerParamChanged() {
    // Go through each of the model layers and propagate shapes.
    let lastOutputShape = inputShape;
    for (let i = 0; i < hiddenLayers.length; i++) {
        lastOutputShape = hiddenLayers[i].setInputShape(lastOutputShape);
    }
    validateModel();

    if (isValid) {
        createModel();
        // startInference();
    }
}

function removeAllLayers() {
    for (let i = 0; i < hiddenLayers.length; i++) {
        layersContainer.removeChild(hiddenLayers[i]);
    }
    hiddenLayers = [];
    layerParamChanged();
}

function updateSelectedModel(modelName) {
    removeAllLayers();
    if (modelName === 'Custom') {
        // TODO(nsthorat): Remember the custom layers.
        return;
    }

    loadModelFromPath(xhrDatasetConfigs[selectedDatasetName].modelConfigs[modelName].path);
}


function addLayer() {
    const modelLayer = new ModelLayer(); //document.createElement('model-layer');
    // var t = document.createTextNode("hidden-layer in:" + labelShapeDisplay);
    // modelLayer.appendChild(t)
    // modelLayer.appendChild(document.createElement("br"));
    // modelLayer.className = 'model-layer';


    const lastHiddenLayer = hiddenLayers[this.hiddenLayers.length - 1];
    const lastOutputShape = lastHiddenLayer != null ?
        lastHiddenLayer.getOutputShape() :
        inputShape;
    hiddenLayers.push(modelLayer);
    modelLayer.initialize(window, lastOutputShape);

    layersContainer.appendChild(modelLayer.paramContainer);
    return modelLayer;
}

function loadModelFromJson(modelJson) {
    let lastOutputShape = inputShape;

    const layerBuilders = JSON.parse(modelJson);
    for (let i = 0; i < layerBuilders.length; i++) {
        const modelLayer = addLayer();
        modelLayer.loadParamsFromLayerBuilder(lastOutputShape, layerBuilders[i]);
        lastOutputShape = hiddenLayers[i].setInputShape(lastOutputShape);
    }
    validateModel();
}

function loadModelFromPath(modelPath) {
    const xhr = new XMLHttpRequest();
    xhr.open('GET', modelPath);

    xhr.onload = () => {
        loadModelFromJson(xhr.responseText);
    };
    xhr.onerror = (error) => {
        throw new Error(`Model could not be fetched from ${modelPath}: ${error}`);
    };
    xhr.send();
}

function displayBatchesTrained(totalBatchesTrained) {
    examplesTrained = batchSize * totalBatchesTrained;
}

function displayCost(avgCost) {
    costChartData.push({
        x: graphRunner.getTotalBatchesTrained(),
        y: avgCost.get()
    });
    costChart.update();
}

function displayAccuracy(accuracy) {
    accuracyChartData.push({
        x: graphRunner.getTotalBatchesTrained(),
        y: accuracy.get() * 100
    });
    accuracyChart.update();
}


function displayInferenceExamplesOutput(
    inputFeeds, inferenceOutputs) {
    let images = [];
    const logits = [];
    const labels = [];
    for (let i = 0; i < inputFeeds.length; i++) {
        images.push(inputFeeds[i][IMAGE_DATA_INDEX].data);
        labels.push(inputFeeds[i][LABEL_DATA_INDEX].data);
        logits.push(inferenceOutputs[i]);
    }

    images =
        dataSet.unnormalizeExamples(images, IMAGE_DATA_INDEX);

    // Draw the images.
    for (let i = 0; i < inputFeeds.length; i++) {
        inputNDArrayVisualizers[i].saveImageDataFromNDArray(images[i]);
    }

    // Draw the logits.
    for (let i = 0; i < inputFeeds.length; i++) {
        const softmaxLogits = math.softmax(logits[i]);

        outputNDArrayVisualizers[i].drawLogits(
            softmaxLogits, labels[i],
            xhrDatasetConfigs[selectedDatasetName].labelClassNames);
        inputNDArrayVisualizers[i].draw();

        softmaxLogits.dispose();
    }
}

function smoothExamplesPerSec(
    lastExamplesPerSec, nextExamplesPerSec) {
    return Number((EXAMPLE_SEC_STAT_SMOOTHING_FACTOR * lastExamplesPerSec +
            (1 - EXAMPLE_SEC_STAT_SMOOTHING_FACTOR) * nextExamplesPerSec)
        .toPrecision(3));
}

function displayInferenceExamplesPerSec(examplesPerSec) {
    inferencesPerSec =
        smoothExamplesPerSec(inferencesPerSec, examplesPerSec);
    inferenceDuration = Number((1000 / examplesPerSec).toPrecision(3));
}


function displayExamplesPerSec(examplesPerSec) {
    examplesPerSecChartData.push({
        x: graphRunner.getTotalBatchesTrained(),
        y: examplesPerSec
    });
    examplesPerSecChart.update();
    examplesPerSec =
        smoothExamplesPerSec(examplesPerSec, examplesPerSec);
}

function run() {
    var mathGPU = new NDArrayMathGPU();
    var mathCPU = new NDArrayMathCPU();
    math = mathGPU;

    const eventObserver = {
        batchesTrainedCallback: (batchesTrained) => displayBatchesTrained(batchesTrained),
        avgCostCallback: (avgCost) => displayCost(avgCost),
        metricCallback: (metric) => displayAccuracy(metric),
        inferenceExamplesCallback: (inputFeeds, inferenceOutputs) => displayInferenceExamplesOutput(inputFeeds, inferenceOutputs),
        inferenceExamplesPerSecCallback: (examplesPerSec) => displayInferenceExamplesPerSec(examplesPerSec),
        trainExamplesPerSecCallback: (examplesPerSec) => displayExamplesPerSec(examplesPerSec),
        totalTimeCallback: (totalTimeSec) => totalTimeSec = totalTimeSec.toFixed(1),
    };
    graphRunner = new GraphRunner(math, session, eventObserver);
    optimizer = new MomentumOptimizer(learningRate, momentum);

    // Set up datasets.
    populateDatasets();

    document.querySelector('#dataset-dropdown').addEventListener(
        // tslint:disable-next-line:no-any
        'iron-activate', (event) => {
            // Update the dataset.
            const datasetName = event.detail.selected;
            updateSelectedDataset(datasetName);

            // TODO(nsthorat): Remember the last model used for each dataset.
            removeAllLayers();
        });
    document.querySelector('#model-dropdown').addEventListener(
        // tslint:disable-next-line:no-any
        'iron-activate', (event) => {
            // Update the model.
            const modelName = event.detail.selected;
            updateSelectedModel(modelName);
        });

    {
        const normalizationDropdown =
            document.querySelector('#normalization-dropdown');
        // tslint:disable-next-line:no-any
        normalizationDropdown.addEventListener('iron-activate', (event) => {
            const selectedNormalizationOption = event.detail.selected;
            applyNormalization(selectedNormalizationOption);
            setupDatasetStats();
        });
    }
    document.querySelector('#optimizer-dropdown').addEventListener('iron-activate', (event) => {
        // Activate, deactivate hyper parameter inputs.
        refreshHyperParamRequirements(event.detail.selected);
    });

    learningRate = 0.1;
    momentum = 0.1;
    var needMomentum = true;
    gamma = 0.1;
    var needGamma = false;
    beta1 = 0.9;
    var needBeta1 = false;
    beta2 = 0.999;
    var needBeta2 = false;
    batchSize = 64;

    // Default optimizer is momentum
    selectedOptimizerName = 'momentum';
    var optimizerNames = ['sgd', 'momentum', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax'];

    applicationState = ApplicationState.IDLE;
    // var loadedWeights = null;
    // var modelInitialized = false;
    // var showTrainStats = false;
    // var showDatasetStats = false;

    // const addButton = document.querySelector('#add-layer');
    // addButton.addEventListener('click', () => addLayer());

    // const downloadModelButton = document.querySelector('#download-model');
    // downloadModelButton.addEventListener('click', () => downloadModel());
    // const uploadModelButton = document.querySelector('#upload-model');
    // uploadModelButton.addEventListener('click', () => uploadModel());
    // setupUploadModelButton();

    // const uploadWeightsButton = document.querySelector('#upload-weights');
    // uploadWeightsButton.addEventListener('click', () => uploadWeights());
    // setupUploadWeightsButton();

    // const stopButton = document.querySelector('#stop');
    // stopButton.addEventListener('click', () => {
    //     applicationState = ApplicationState.IDLE;
    //     graphRunner.stopTraining();
    // });

    // trainButton = document.querySelector('#train');
    // trainButton.addEventListener('click', () => {
    //     createModel();
    //     startTraining();
    // });

    document.querySelector('#environment-dropdown').addEventListener('change', (event) => {
        math = (event.target).active ? mathGPU : mathCPU;
        graphRunner.setMath(math);
    });

    hiddenLayers = [];
    examplesPerSec = 0;
    inferencesPerSec = 0;
}

var infer_request = null;
var btn_infer = document.getElementById('buttoninfer');
var infer_paused = true;
btn_infer.addEventListener('click', () => {

    infer_paused = !infer_paused;
    if (infer_paused) {
        btn_infer.value = 'Start Inferring';
        if (graphRunner != null) {
            graphRunner.stopInferring();
        }


    } else {

        infer_request = true;
        btn_infer.value = 'Pause Inferring';


    }
});


var train_request = null;
var btn_train = document.getElementById('buttontrain');
var train_paused = true;
btn_train.addEventListener('click', () => {
    train_paused = !train_paused;

    if (train_paused) {
        if (graphRunner != null) {
            graphRunner.stopTraining();
        }
        btn_train.value = 'Start Training';

    } else {

        train_request = true;

        btn_train.value = 'Pause Training';

    }
});


function monitor() {

    if (modelInitialized == false) {

        btn_infer.className = 'btn btn-info btn-md';
        btn_infer.disabled = true;
        btn_infer.value = 'Initializing Model ...'
        // btn_train.disabled = true;
        btn_train.style.visibility = 'hidden';

    } else {
        if (isValid) {

            btn_infer.className = 'btn btn-primary btn-md';
            btn_infer.disabled = false;
            btn_train.style.visibility = 'visible';

            if (infer_paused) {
                btn_infer.value = 'Start Infering'
            } else {
                btn_infer.value = 'Stop Infering'
            }

            if (train_paused) {
                btn_train.value = 'Start Training'
            } else {
                btn_train.value = 'Stop Training'
            }

            if (train_request) {
                train_request = false;
                // createModel();
                startTraining();
            }

            if (infer_request) {
                infer_request = false;
                // createModel();
                startInference();
            }

        } else {
            btn_infer.className = 'btn btn-danger btn-md';
            btn_infer.disabled = true;
            btn_infer.value = 'Model not valid'
            // btn_train.disabled = true;
            btn_train.style.visibility = 'hidden';
        }
    }

    setTimeout(function () {
        monitor();
    }, 0);
}


function start() {


    supported = detect_support();

    if (supported) {
        console.log('device & webgl supported');
        btn_infer.disabled = false;
        btn_train.disabled = false;
    } else {
        console.log('device/webgl not supported')
        btn_infer.disabled = true;
        btn_train.disabled = true;
    }

    setTimeout(function () {

        run();

        monitor();


    }, 0);

}