const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');
const load_model_from_txt = require('./model_from_txt.js')

const fs = require('fs');

const model_path = './ckpt/rnn_based_16';
const txt_path = './txt_model/model_16.txt';

async function loadModel(path) {
    const model = await tf.loadLayersModel(`file://${path}`);
    return model;
}

function loadModelForWindows() {
    const model = load_model_from_txt();
    return model;
}

// Currently for windows
function infer(img_path) {
    opencv
}