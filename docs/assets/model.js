class Model {
  constructor() { }

  async load() {
      this.session = await ort.InferenceSession.create('model/cnn_74.optimized.onnx');
  }

  /**
   *
   * @param {Float32Array} rgb_input
   * @returns probablities of classes
   */
  async infer(rgb_input) {
      const dims = [1, 3, 32, 32];
      const tensor = new ort.Tensor('float32', rgb_input, dims);
      const result = await this.session.run({ 'input.1': tensor });
      const probs = Array.from(result[Object.keys(result)].cpuData);
      return probs.map((x, i) => ({ logit: x, probability: Math.exp(x), index: i }));
  }
}
