var dl = deeplearn;

var Array3D = dl.Array3D;
var Array4D = dl.Array4D;
var Array1D = dl.Array1D;
var ENV = dl.ENV;

var GPGPUContext = dl.GPGPUContext;
var gpgpu_util = dl.gpgpu_util;
var render_ndarray_gpu_util = dl.render_ndarray_gpu_util;
var NDArrayMathCPU = dl.NDArrayMathCPU;
var NDArrayMathGPU = dl.NDArrayMathGPU;


// Initialize deeplearn.js stuff
canvas = document.querySelector('#imageCanvas');
gl = gpgpu_util.createWebGLContext(canvas);
gpgpu = new GPGPUContext(gl);
math = new NDArrayMathGPU(gpgpu);
mathCPU = new NDArrayMathCPU();

// Initialize polymer properties
var ApplicationState = {
    IDLE: 1,
    TRAINING: 2
};
applicationState = ApplicationState.IDLE;
status = '';

// Retrieve DOM for images
contentImgElement =
    document.querySelector('#contentImg');
styleImgElement =
    document.querySelector('#styleImg');

// Render DOM for images
const CONTENT_NAMES = ['stata', 'face', 'diana', 'Upload from file'];
const STYLE_MAPPINGS = {
    'Udnie, Francis Picabia': 'udnie',
    'The Scream, Edvard Munch': 'scream',
    'La Muse, Pablo Picasso': 'la_muse',
    'Rain Princess, Leonid Afremov': 'rain_princess',
    'The Wave, Katsushika Hokusai': 'wave',
    'The Wreck of the Minotaur, J.M.W. Turner': 'wreck'
};
const STYLE_NAMES = Object.keys(STYLE_MAPPINGS);

contentNames = CONTENT_NAMES;
selectedContentName = 'stata';
contentImgElement.src = 'deeplearnjs/images/stata.jpg';
contentImgElement.height = 250;

styleNames = STYLE_NAMES;
selectedStyleName = 'Udnie, Francis Picabia';
styleImgElement.src = 'deeplearnjs/images/udnie.jpg';
styleImgElement.height = 250;

transformNet = new TransformNet(math,
    STYLE_MAPPINGS[selectedStyleName]);


function initWebcamVariables() {
    camDialog = document.querySelector('#webcam-dialog');
    webcamVideoElement = document.querySelector('#webcamVideo');
    takePicButton = document.querySelector('#takePicButton');
    closeModal = document.querySelector('#closeModal');

    // Check if webcam is even available
    // tslint:disable-next-line:no-any
    const navigatorAny = navigator;
    navigator.getUserMedia = navigator.getUserMedia ||
        navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
        navigatorAny.msGetUserMedia;
    if (navigator.getUserMedia) {
        const contentNames = CONTENT_NAMES.slice();
        contentNames.unshift('Use webcam');
        contentNames = contentNames;
    }

    closeModal.addEventListener('click', () => {
        stream.getTracks()[0].stop();
    });

    takePicButton.addEventListener('click', () => {
        const hiddenCanvas =
            querySelector('#hiddenCanvas');
        const hiddenContext =
            hiddenCanvas.getContext('2d');
        hiddenCanvas.width = webcamVideoElement.width;
        hiddenCanvas.height = webcamVideoElement.height;
        hiddenContext.drawImage(webcamVideoElement, 0, 0,
            hiddenCanvas.width, hiddenCanvas.height);
        const imageDataURL = hiddenCanvas.toDataURL('image/jpg');
        contentImgElement.src = imageDataURL;
        stream.getTracks()[0].stop();
    });
}

initWebcamVariables();

// tslint:disable-next-line:no-any
sizeSlider = document.querySelector('#sizeSlider');
sizeSlider.addEventListener('immediate-value-change',
    // tslint:disable-next-line:no-any
    (event) => {
        styleImgElement.height = sizeSlider.immediateValue;
        contentImgElement.height = sizeSlider.immediateValue;
    });
// tslint:disable-next-line:no-any
sizeSlider.addEventListener('change', (event) => {
    styleImgElement.height = sizeSlider.immediateValue;
    contentImgElement.height = sizeSlider.immediateValue;
});

fileSelect = document.querySelector('#fileSelect');
// tslint:disable-next-line:no-any
fileSelect.addEventListener('change', (event) => {
    const f = event.target.files[0];
    const fileReader = new FileReader();
    fileReader.onload = ((e) => {
        const target = e.target;
        contentImgElement.src = target.result;
    });
    fileReader.readAsDataURL(f);
    fileSelect.value = '';
});

// Add listener to drop downs
const contentDropdown = document.querySelector('#content-dropdown');
// tslint:disable-next-line:no-any
contentDropdown.addEventListener('iron-activate', (event) => {
    const selected = event.detail.selected;
    if (selected === 'Use webcam') {
        openWebcamModal();
    } else if (selected === 'Upload from file') {
        fileSelect.click();
    } else {
        contentImgElement.src = 'images/' + selected + '.jpg';
    }
});

const styleDropdown = document.querySelector('#style-dropdown');
// tslint:disable-next-line:no-any
styleDropdown.addEventListener('iron-activate', (event) => {
    styleImgElement.src =
        'images/' + STYLE_MAPPINGS[event.detail.selected] + '.jpg';
});

// Add listener to start
startButton = document.querySelector('#start');
startButton.addEventListener('click', () => {
    (document.querySelector('#load-error-message')).style.display =
        'none';
    startButton.textContent =
        'Starting style transfer.. Downloading + running model';
    startButton.disabled = true;
    transformNet.setStyle(STYLE_MAPPINGS[selectedStyleName]);

    transformNet.load()
        .then(() => {
            startButton.textContent = 'Processing image';
            runInference();
            startButton.textContent = 'Start Style Transfer';
            startButton.disabled = false;
        })
        .catch((error) => {
            console.log(error);
            startButton.textContent = 'Start Style Transfer';
            startButton.disabled = false;
            const errMessage =
                document.querySelector('#load-error-message');
            errMessage.textContent = error;
            errMessage.style.display = 'block';
        });
});


var keep = dl.keep;
var track = dl.track;
var gpgpu = dl.gpgpu;

function runInference() {
    math.scope((keep, track) => {

        const preprocessed = track(Array3D.fromPixels(contentImgElement));

        const inferenceResult = transformNet.predict(preprocessed);

        setCanvasShape(inferenceResult.shape);
        renderShader = render_ndarray_gpu_util.getRenderRGBShader(
            gpgpu, inferenceResult.shape[1]);
        render_ndarray_gpu_util.renderToCanvas(
            gpgpu, renderShader, inferenceResult.getTexture());
    });
}

function setCanvasShape(shape) {
    canvas.width = shape[1];
    canvas.height = shape[0];
    if (shape[1] > shape[0]) {
        canvas.style.width = '500px';
        canvas.style.height = (shape[0] / shape[1] * 500).toString() + 'px';
    } else {
        canvas.style.height = '500px';
        canvas.style.width = (shape[1] / shape[0] * 500).toString() + 'px';
    }
}