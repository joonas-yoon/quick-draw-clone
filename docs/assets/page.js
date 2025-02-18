const SHOW_TOP_K = 20;

const body = document.body;
const resultBox = document.getElementById('result');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const cursor = {
    isDrawing: false,
    hasChange: false,
    px: null,
    py: null,
};
const STROKE_SIZE = 3;
const cropBox = {
    isValid: false,
    top: Infinity,
    left: Infinity,
    right: -Infinity,
    bottom: -Infinity
};
let labels;
let model;

const canvasCopy = document.createElement("canvas");
const copyContext = canvasCopy.getContext("2d");

let cpuActions = [];
let candidatesPool;
const gameState = {
    answer: null,
    isRunning: false,
};

const cpuTextElement = document.getElementById('cpu-text');
const startPageElement = document.getElementById('round-start-page');
const answerText = document.getElementById('answer');

function createCpuMessageAction(message) {
    return {
        answer: null,
        message,
        id: null,
    };
}

function createCpuAnswerAction(answer, keywordId) {
    const randomTexts = [
        `이제 알겠어요, <strong>${answer}</strong> 인가요?`,
        `정답! <strong>${answer}</strong> 이죠?`,
        `혹시 <strong>${answer}</strong> 맞나요?`,
        `...<strong>${answer}</strong>!`,
        `<strong>${answer}</strong>!`,
    ];
    return {
        answer,
        message: randomChoice(randomTexts),
        id: keywordId,
    };
}

function isDebug() {
    return (body.className || "").indexOf("debug") !== -1;
}

window.onload = async () => {
    resize();

    const { label: _labels } = await fetch("https://cdn.jsdelivr.net/gh/joonas-yoon/quick-draw-clone/docs/assets/labels.json")
        .then(response => response.json());
    labels = _labels;

    console.log('labels', labels);

    candidatesPool = createNewPool(labels.length);

    choiceAnswerForQuiz();
    setupUI();
    setupCanvas();
};

function setupUI() {
    document.getElementById('start-button').addEventListener('click', (evt) => {
        evt.preventDefault();
        startGame();
    });
    document.getElementById('restart-button').addEventListener('click', (evt) => {
        evt.preventDefault();
        endGame();
    });
    document.getElementById('erase-button').addEventListener('click', (evt) => {
        evt.preventDefault();
        resize();
        clearCanvas();
    });
    const debugButton = document.getElementById('debug-button');
    debugButton.addEventListener('click', (evt) => {
        evt.preventDefault();
        if (!!debugButton.getAttribute('data-active')) {
            debugButton.removeAttribute('data-active');
            body.className = '';
        } else {
            debugButton.setAttribute('data-active', true);
            body.className = 'debug';
        }
    });
}

async function setupCanvas() {
    canvas.addEventListener('mousedown', () => cursor.isDrawing = true);
    canvas.addEventListener('mouseup', () => {
        cursor.isDrawing = false;
        cursor.px = null;
        cursor.py = null;
        if (isDebug()) {
            console.log(cropBox);
            // drawBox(cropBox);
        }
    });
    canvas.addEventListener('mousemove', (evt) => {
        if (!cursor.isDrawing) return;
        const { clientX, clientY } = evt;
        stroke(clientX, clientY, cursor.px, cursor.py, STROKE_SIZE);
        cursor.px = clientX;
        cursor.py = clientY;
        cursor.hasChange = true;
        cropBox.isValid = true;
        const PADDING = 5;
        cropBox.top = Math.min(cropBox.top, clientY - PADDING);
        cropBox.bottom = Math.max(cropBox.bottom, clientY + PADDING);
        cropBox.left = Math.min(cropBox.left, clientX - PADDING);
        cropBox.right = Math.max(cropBox.right, clientX + PADDING);
    });
    canvas.addEventListener('click', (evt) => {
        drawPoint(evt.clientX, evt.clientY, STROKE_SIZE);
    });

    canvasCopy.width = 32;
    canvasCopy.height = 32;

    const startTime = now();
    model = new Model();
    await model.load();
    console.log('loading time:', timeDelta(startTime, now()), 'ms');
}

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function startGame() {
    startPageElement.classList.add('hide');
    gameState.isRunning = true;

    // fetch from queue to infer
    createRandomInterval(() => {
        if (!gameState.isRunning) {
            return false;
        }
        if (cursor.hasChange) {
            inference();
            cursor.hasChange = false;
        }
        return true;
    }, 600, 1500);

    // cpu keep talking randomly
    createRandomInterval(() => {
        if (!gameState.isRunning) {
            return false;
        }
        const randomTexts = [
            '...',
            '음...',
            '음... 뭘까요?',
            '조금만 더 그려주세요',
            '더 그려주시겠어요?',
        ];
        cpuActions.push(createCpuMessageAction(randomChoice(randomTexts)));
        return true;
    }, 3000, 4000);

    // cpu does action and popped from queue
    createRandomInterval(() => {
        if (!gameState.isRunning) {
            return false;
        }
        if (cpuActions.length < 1) {
            return true;
        }
        const {answer, message} = cpuActions[0];
        cpuActions = cpuActions.slice(1); // pop
        cpuTextElement.innerHTML = message;
        if (answer !== null) {
            console.log('GUESS WHAT:', answer);
            if (answer === gameState.answer) {
                cpuTextElement.innerHTML = `알겠어요! 정답은 <strong></strong>!`;
                endGame();
            }
        }
        return true;
    }, 2000, 4000);

    // first cpu message
    cpuActions.push(createCpuMessageAction('...'));
}

function endGame() {
    startPageElement.classList.remove('hide');
    gameState.isRunning = false;
    setTimeout(() => {
        clearCanvas();
        choiceAnswerForQuiz();
    }, 2000);
}

function choiceAnswerForQuiz() {
    gameState.answer = randomChoice(labels)['ko'];
    answerText.textContent = gameState.answer;
}

function createRandomInterval(func, minMS, maxMS) {
    let timer = null;
    function _innerLoop() {
        if (timer !== null) {
            clearTimeout(timer);
        }
        if (func()) {
            timer = setTimeout(_innerLoop, minMS + Math.random() * (maxMS - minMS));
        }
    }
    _innerLoop();
}

window.onresize = resize;

function resize() {
    // resize canvas
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    // reset crop box
    cropBox.top = Infinity;
    cropBox.left = Infinity;
    cropBox.right = -Infinity;
    cropBox.bottom = -Infinity;
}

/**
 * @param image Float32Array
 **/
function normalize(image) {
    const [mean, std] = [0.5, 0.5];
    return image.map(i => (i - mean) / std);
}

function inference() {
    if (model == null || labels == null) {
        console.error('Resources not ready to run yet');
        return;
    }
    const { top, left, right, bottom } = cropBox;
    const oImgWidth = right - left;
    const oImgHeight = bottom - top;

    if (!isFinite(oImgWidth) || !isFinite(oImgHeight)) {
        console.log('No information to inference right now');
        return;
    }

    const imageProcessStartTime = now();
    const imageDrawData = ctx.getImageData(left, top, oImgWidth, oImgHeight);
    const imageDraw = imagedataToImage(imageDrawData);
    imageDraw.onload = async () => {
        if (isDebug()) {
            document.getElementById('preview').src = imageDraw.src;
            console.log(oImgWidth, oImgHeight);
        }

        // copy and resize image to (32, 32)
        copyContext.clearRect(0, 0, 32, 32);
        copyContext.drawImage(imageDraw, 0, 0, oImgWidth, oImgHeight, 0, 0, 32, 32);

        if (isDebug()) {
            const tempCtx = document.getElementById('canvas3').getContext('2d');
            tempCtx.clearRect(0, 0, 32, 32);
            tempCtx.drawImage(imageDraw, 0, 0, oImgWidth, oImgHeight, 0, 0, 32, 32);
        }

        // RGBA image in range of [0, 255] to RGB image in range of [0.0, 1.0]
        const imageResize = copyContext.getImageData(0, 0, 32, 32);
        const imageResizeRGB = imageResize.data.filter((value, index, _) => index % 4 != 0);
        const input = new Float32Array(imageResizeRGB).map(i => i / 255.);
        if (isDebug()) {
            console.log('input image:', imageResize.data, input);
        }
        console.info('image processed in', timeDelta(imageProcessStartTime, now()), 'ms');

        // normalize image and infer
        const modelInferStartTime = now();
        const output = await model.infer(normalize(input));
        output.sort((a, b) => b.logit - a.logit);
        if (isDebug()) {
            console.log('inference', output);
        }
        console.info('inferenced in', timeDelta(modelInferStartTime, now()), 'ms');

        pickAnswer(output);

        // show result on HUD
        showResultHTML(output.slice(0, SHOW_TOP_K).map(e => ({
            ...e, label: labels[e.index]
        })));
    };

    function pickAnswer(predictProbs) {
        // if already too many in queue, refresh answers
        const readyAnswers = cpuActions.filter(({id}) => id != null);
        console.log('=======================', { readyAnswers });
        if (readyAnswers.length > 2) {
            for (const { id } of readyAnswers) {
                candidatesPool.add(id);
            }
            // clear
            cpuActions = [];
        }

        // guess
        const guessIndex = pickFirst(predictProbs, candidatesPool);
        if (guessIndex !== -1) {
            const labelString = labels[guessIndex]['ko'];
            cpuActions.push(createCpuAnswerAction(labelString, guessIndex));
            candidatesPool.delete(guessIndex);
        }
        console.log('cpuActionQueue', cpuActions);
    }
}

function dist(x1, y1, x2, y2) {
    const dx = x1 - (x2 == null ? x1 : x2);
    const dy = y1 - (y2 == null ? y1 : y2);
    return dx * dx + dy * dy;
}

function drawPoint(x, y, rad) {
    ctx.beginPath();
    ctx.fillStyle = '#222222ff';
    ctx.arc(x, y, rad, 0, 2 * Math.PI);
    ctx.fill();
    ctx.closePath();
}

function drawLine(x, y, x2, y2, width) {
    ctx.beginPath();
    ctx.strokeStyle = '#222222ff';
    ctx.lineWidth = width;
    ctx.lineCap = 'round';
    ctx.moveTo(x, y);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.closePath();
}

function drawBox({ top, left, right, bottom }) {
    if (!isFinite(top) || !isFinite(left) || !isFinite(right) || !isFinite(bottom)) return;
    ctx.beginPath();
    ctx.rect(left, top, right - left, bottom - top);
    ctx.strokeStyle = '#ff666699';
    ctx.stroke();
    ctx.closePath();
}

function stroke(x, y, px, py, rad) {
    const d = dist(x, y, px, py);
    if (d > rad * rad) {
        drawLine(x, y, px, py, rad * 2);
    } else {
        drawPoint(x, y, rad);
    }
}

function now() {
    return (new Date()).getTime();
}

function timeDelta(t1, t2) {
    return t2 - t1;
}

function randomChoice(array) {
    const index = Math.floor(Math.random() * array.length);
    return array[index];
}

function showResultHTML(results) {
    resultBox.innerHTML = '';
    console.log('candidatesPool', candidatesPool);
    console.log('results', results);
    for (const { label, probability, index } of results) {
        const item = document.createElement('li');
        const alreadyMentioned = !candidatesPool.has(index);
        const opacity = Math.max(0.15, Math.sqrt(Math.sqrt(probability)));
        item.style = `opacity: ${opacity}`;
        item.className = alreadyMentioned ? 'strike' : '';
        item.textContent = `${label['ko']} (${(probability * 100).toFixed(2)}%)`;
        resultBox.appendChild(item);
    }
}

// https://stackoverflow.com/a/41229530/13677554
function imagedataToImage(imagedata) {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    canvas.width = imagedata.width;
    canvas.height = imagedata.height;
    context.putImageData(imagedata, 0, 0);
    const image = new Image();
    image.src = canvas.toDataURL();
    delete canvas;
    return image;
}

function _createLengthedArray(length) {
    return Array.from(new Array(length));
}

function createNewPool(length) {
    return new Set(_createLengthedArray(length).map((x, i) => i));
}

function pickFirst(predictProbs, filter) {
    for (const {probability, index} of predictProbs) {
        if (filter.has(index)) {
            return index;
        }
    }
    return -1;
}
