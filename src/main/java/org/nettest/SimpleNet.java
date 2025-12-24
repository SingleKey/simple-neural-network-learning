package org.nettest;

import java.util.Arrays;
import java.util.Random;

public class SimpleNet {

    /**
     * 固定输入层、隐藏层、输出层大小（核心数据）
     */
    private static final int INPUT_LAYER_SIZE = 9;
    private static final int HIDDEN_LAYER_SIZE = 5;
    private static final int OUTPUT_LAYER_SIZE = 4;

    /**
     * 学习率
     */
    private static final double LEARNING_RATE = 0.1;

    /**
     * 输入层（临时变量，不参与持久化）
     */
    private int[] inputLayer = new int[INPUT_LAYER_SIZE];

    /**
     * 训练时使用的答案（临时变量，不参与持久化）
     */
    private int answer;

    /**
     * 权重层，数量是输入层数量 * 计算结果层数量（核心数据）
     */
    private double[] inputAndHideWeightLayer = new double[INPUT_LAYER_SIZE * HIDDEN_LAYER_SIZE];

    /**
     * 偏导数，用于平移函数（核心数据）
     */
    private double[] hiddenLayerBiasArray = new double[HIDDEN_LAYER_SIZE];

    /**
     * 输入层和权重层的计算结果层，一般输出层的结果数量要比输入层更少（临时变量，用于存储计算结果，不参与持久化）
     */
    private double[] hideLayer = new double[HIDDEN_LAYER_SIZE];

    /**
     * 权重层，数量是输入层数量 * 输入结果层数量（核心数据）
     */
    private double[] hideAndOutputWeightLayer = new double[HIDDEN_LAYER_SIZE * OUTPUT_LAYER_SIZE];

    /**
     * 偏导数，用于平移函数（核心数据）
     */
    private double[] outputLayerBiasArray = new double[OUTPUT_LAYER_SIZE];

    /**
     * 输出层，一般输出层的结果数量要比输入层更少（临时变量，用于存储计算结果，不参与持久化）
     */
    private double[] outputLayer = new double[OUTPUT_LAYER_SIZE];


    /**
     * 8、输入预测数据
     *
     * @param dataArray
     */
    public void setInput(int... dataArray) {
        if (dataArray.length != 9) {
            throw new RuntimeException("输入数据量必须为9个！");
        }
        for (int i = 0; i < dataArray.length; i++) {
            this.inputLayer[i] = dataArray[i];
        }
    }

    /**
     * 设置答案
     */
    public void setAnswer(int answer) {
        this.answer = answer;
    }

    private Random random = new Random(System.currentTimeMillis());

    /**
     * 0、初始化或者重置权重层
     */
    public void initWeightLayer() {
        randomValue(inputAndHideWeightLayer);
        randomValue(hideAndOutputWeightLayer);
        randomValue(hiddenLayerBiasArray);
        randomValue(outputLayerBiasArray);
    }

    /**
     * 随机化数组值
     * @param arr
     */
    private void randomValue(double[] arr) {
        for (int i = 0; i < arr.length; i++) {
            // 初始化为-1到1范围内的随机数
            arr[i] = 2.0 * random.nextDouble() - 1.0;
        }
    }

    /**
     * 1、输入训练数据
     */
    public void addTrainData(int answer, int... dataArray) {
        setInput(dataArray);
        if (answer >= OUTPUT_LAYER_SIZE || answer < 0) {
            throw new RuntimeException("输入的答案必须为0-" + OUTPUT_LAYER_SIZE + "的范围！");
        }
    }

    /**
     * 2、训练
     */

    public void training(int answer, int...input) {
        setAnswer(answer);
        setInput(input);
        training();
    }

    /**
     * 训练方法
     *
     * 神经网络推导过程
     *
     * 前向传播其实是根据以下的顺序进行计算的
     * a → [a·w₁ + b₁] → m → sigmoid → h → [h·w₂ + b₂] → z → softmax(z) → o
     * 在反向传播中，需要通过链式法则对当前函数的组成部分进行求导，因为函数之间的关系就是组成关系，需要找该函数的组成部分，就需要使用链式法则公式dy/dx = dy/du*du/dx，
     *
     * 例如：a * b = c，dc/db = a，读作c对b的偏导数等于a，dc/da = b，c对a的偏导数等于b，可以理解为，函数c是由函数a和函数b组成。
     *
     * 因为o函数的直接组成部分只有z，所以需要对z进行求导，z经过softmax函数的计算，所以求z的导数就是loss函数，设真实标签为y，即dL/dz = (o - y)，
     * 因为整个网络结构都是一个完整的算式，所以可以根据链式法则，找到对应函数的组成部分，
     * 比如h * w₂ + b₂ = z，z函数的组成部分是h和w，而此处的b在函数图像中用于平移，并不影响函数图像的形状，所以这里的b导数是1，但是一般不写，
     *
     * 根据h * w₂ + b₂ = z可以得出对应的求导表达式dz/dw₂ = h，dz/b₂ = 1，dz/dh = w₂，因为此处局部表达式的h和w₂前面没有数，所以不再分解
     * 因为要修改w₂，可以得出dL/dw₂ = dL/dz * dz/dw₂ * 1，但是一般不写1，所以简化为dL/dw₂ = dL/dz * dz/dw₂，
     * 因为dz/dw₂ = h，所以简化为dL/dw₂ = dL/dz * h，与学习率相乘，再赋以负号表示反方向调节，让w₂加上这个数进行调节就完成对w₂的调整，
     * 修改b₂的式子为dL/db₂ = dL/dz * dz/db₂，因为dz/b₂ = 1，所以dL/db₂ = dL/dz，与学习率相乘，再赋以负号表示反方向调节，让b₂加上这个数进行调节就完成对b₂的调整，
     *
     * 对sigmoid求导得dh/dm = h*(1 - h)，根据a * w₁ + b₁ = m，得dm/dw₁ = a, dm/b₁ = 1，需要修改w₁，
     * 所以求dL/dw₁：dL/dw₁ = dL/dh * dh/dm * dm/dw₁;
     * 求dL/db₁: dL/db₁ = dL/dh * dh/dm，同样将结果与学习率相乘，再赋以负号调节即可。
     */
    private void training() {
        //=============== 前向传播 =================
        // 等待更新的隐藏层位置游标
        int inputAndHideWeightLayerPoint = 0;
        // 当前输入层
        double inputNum = 0.0;
        // 暂存计算结果
        double calc = 0.0;
        for (int hiddenPoint = 0; hiddenPoint < HIDDEN_LAYER_SIZE; hiddenPoint++) {
            for (int inputPoint = 0; inputPoint < INPUT_LAYER_SIZE; inputPoint++) {
                inputNum = inputLayer[inputPoint];
                // 这里计算要更新的层位置，比如0-100，每层10个房间，第0层的十位数就是0，加上个位数inputPoint就是对应房间号码
                inputAndHideWeightLayerPoint = (hiddenPoint * INPUT_LAYER_SIZE) + inputPoint;
                // 计算权重
                calc = calc + (inputNum * inputAndHideWeightLayer[inputAndHideWeightLayerPoint]);
            }
            // 添加偏导数（用于平移函数图像）
            calc = calc + hiddenLayerBiasArray[hiddenPoint];
            // 计算完成，更新到hiddenLayer中
            hideLayer[hiddenPoint] = sigmoid(calc);
            // 重置计算缓存
            calc = 0.0;
        }

        // 等待更新的隐藏层位置游标
        int hideAndOutputWeightLayerPoint = 0;
        // 当前隐藏层
        double hiddenNum = 0.0;
        // 暂存计算结果
        calc = 0.0;
        double[] logits = new double[OUTPUT_LAYER_SIZE];
        for (int outputPoint = 0; outputPoint < OUTPUT_LAYER_SIZE; outputPoint++) {
            for (int hiddenPoint = 0; hiddenPoint < HIDDEN_LAYER_SIZE; hiddenPoint++) {
                hiddenNum = hideLayer[hiddenPoint];
                // 这里计算要更新的层位置，比如0-100，每层10个房间，第0层的十位数就是0，加上个位数inputPoint就是对应房间号码
                hideAndOutputWeightLayerPoint = (outputPoint * HIDDEN_LAYER_SIZE) + hiddenPoint;
                calc = calc + (hiddenNum * hideAndOutputWeightLayer[hideAndOutputWeightLayerPoint]);
            }
            // 添加偏导数（用于平移函数图像）
            calc = calc + outputLayerBiasArray[outputPoint];
            // 计算完成，更新到outputLayer中
            logits[outputPoint] = calc;
            // 重置计算缓存
            calc = 0.0;
        }

        double[] softmax = softmax2();

        //=============== 反向传播 =================
        /*
        反向传播中的链式求导步骤：
        隐藏层到输出层：
        在前向传播过程中，直接可以计算出结果的式子，比如 h * w + b = o，能得出do/dw = h，do/db = 1;
        实际过程中，计算了L = loss(o)，这里的loss(o)在函数表达式中就是do，反向推导出：dL/dw = dL/do * do/dw;
        带入do/dw = h，得dL/dw = dL/do * h，dL/db = dL/do * do/db;
        带入do/db = 1，得dL/db = dL/do * 1;

        输入层到隐藏层：
        a * w + b = h，能得出dh/dw = a，dh/db = 1;
        实际计算中，计算了dz = sigmoid(h)，所以反向推导出：dL/dw = dL/dz * dz/dh * dh/dw;
        带入dh/dw = a得dL/dw = dL/dz * dz/dh * a;
        dL/db = dL/dz * dz/dh * dh/db;
        带入dh/db = 1得dL/db = dL/dz * dz/dh;

         */
        // 反向传播的计算原理是通过上方的调节力度乘以自身的调节力度
        // 如果不存在上方的调节力度，那上方的调节力度就为1
        // 上方的调节力度：如果没有就为1
        // 自身的调节力度：导数
        //  f(a, b) = a * b
        //  → ∂f/∂a = b
        //  → ∂f/∂b = a
        // 导数的公式：
        //   f(a + h, b) - f(a, b) / h
        // = (a + h) * b - (a * b) / h
        // = (ab + hb) - ab / h
        // = hb / h
        // = b
        // 即：f(a, b)在a方向上的导数为b，f(a, b)在b方向上的导数为a
        // 根据此公式
        // 答案构造向量，用于反向传播
        double[] answerVector = answerConstructionVector();

        // =========================
        // 计算交叉熵损失（失效）
//        double crossEntropyLoss = 0.0;
//        for (int outputPoint = 0; outputPoint < OUTPUT_LAYER_SIZE; outputPoint++) {
//            // 加上一个小值防止log(0)
//            crossEntropyLoss += -answerVector[outputPoint] * Math.log(softmax[outputPoint] + 1e-8);
//        }
//        System.out.println("交叉熵损失：" + crossEntropyLoss);
        // =========================

        // 求o的导数，y是正确答案的onehot编码
        // dL/do[k] = softmax[k] - y[k]
        double[] outputError = new double[OUTPUT_LAYER_SIZE];
        for (int outputPoint = 0; outputPoint < OUTPUT_LAYER_SIZE; outputPoint++) {
            outputError[outputPoint] = softmax[outputPoint] - answerVector[outputPoint];
        }

        // 因为前向传播是：o = w2 * h，f(a, b) = a * b，所以∂o/∂w2 = h， ∂o/∂h = w2
        // ∂L/∂w2[j][k] = (∂L/∂o[k]) × (∂o[k]/∂w2[j][k])
        // 因为∂o/∂w2 = h，
        // 所以(∂o[k]/∂w2[j][k]) = h
        // 所以∂L/∂w2[j][k] = (∂L/∂o[k]) × h
        //
        // 求导数w，根据链式法则得出：
        // o=output,w2=hideAndOuputWeightLayer
        // dL/dw2[j][k] = dL/do[k] * do[k]/dh[j]
        for (int outputPoint = 0; outputPoint < OUTPUT_LAYER_SIZE; outputPoint++) {
            for (int hiddenPoint = 0; hiddenPoint < HIDDEN_LAYER_SIZE; hiddenPoint++) {
                // 这里计算要更新的层位置，比如0-100，每层10个房间，第0层的十位数就是0，加上个位数inputPoint就是对应房间号码
                hideAndOutputWeightLayerPoint = (outputPoint * HIDDEN_LAYER_SIZE) + hiddenPoint;
                hideAndOutputWeightLayer[hideAndOutputWeightLayerPoint] += - LEARNING_RATE * outputError[outputPoint] * hideLayer[hiddenPoint];
            }
            // 更新偏置层
            // ∂L/∂bⱼ = ∂L/∂oⱼ × 1
            outputLayerBiasArray[outputPoint] += - LEARNING_RATE * outputError[outputPoint];
        }

        // 计算隐藏层误差
        double[] hiddenError = new double[HIDDEN_LAYER_SIZE];
        for (int hiddenPoint = 0; hiddenPoint < HIDDEN_LAYER_SIZE; hiddenPoint++) {
            double sum = 0;
            for (int outputPoint = 0; outputPoint < OUTPUT_LAYER_SIZE; outputPoint++) {
                // 这里计算要更新的层位置，比如0-100，每层10个房间，第0层的十位数就是0，加上个位数inputPoint就是对应房间号码
                hideAndOutputWeightLayerPoint = (outputPoint * HIDDEN_LAYER_SIZE) + hiddenPoint;
                sum += outputError[outputPoint] * hideAndOutputWeightLayer[hideAndOutputWeightLayerPoint];
            }
            hiddenError[hiddenPoint] = sum * hideLayer[hiddenPoint] * (1 - hideLayer[hiddenPoint]);
        }

        // 更新隐藏层权重，以及隐藏层的偏置参数
        for (int hiddenPoint = 0; hiddenPoint < HIDDEN_LAYER_SIZE; hiddenPoint++) {
            for (int inputPoint = 0; inputPoint < INPUT_LAYER_SIZE; inputPoint++) {
                inputNum = inputLayer[inputPoint];
                // 这里计算要更新的层位置，比如0-100，每层10个房间，第0层的十位数就是0，加上个位数inputPoint就是对应房间号码
                inputAndHideWeightLayerPoint = (hiddenPoint * INPUT_LAYER_SIZE) + inputPoint;
                inputAndHideWeightLayer[inputAndHideWeightLayerPoint] += - LEARNING_RATE * hiddenError[hiddenPoint] * inputNum;
            }
            hiddenLayerBiasArray[hiddenPoint] += - LEARNING_RATE * hiddenError[hiddenPoint];
        }

    }

    /**
     * 3、预测
     * @param input
     */
    public double[] predict(int...input) {
        setInput(input);
        //=============== 前向传播 =================
        // 等待更新的隐藏层位置游标
        int inputAndHideWeightLayerPoint = 0;
        // 当前输入层
        double inputNum = 0.0;
        // 暂存计算结果
        double calc = 0.0;
        for (int hiddenPoint = 0; hiddenPoint < HIDDEN_LAYER_SIZE; hiddenPoint++) {
            for (int inputPoint = 0; inputPoint < INPUT_LAYER_SIZE; inputPoint++) {
                inputNum = inputLayer[inputPoint];
                // 这里计算要更新的层位置，比如0-100，每层10个房间，第0层的十位数就是0，加上个位数inputPoint就是对应房间号码
                inputAndHideWeightLayerPoint = (hiddenPoint * INPUT_LAYER_SIZE) + inputPoint;
                // 计算权重
                calc = calc + (inputNum * inputAndHideWeightLayer[inputAndHideWeightLayerPoint]);
            }
            // 添加偏导数（用于平移函数图像）
            calc = calc + hiddenLayerBiasArray[hiddenPoint];
            // 计算完成，更新到hiddenLayer中
            hideLayer[hiddenPoint] = sigmoid(calc);
            // 重置计算缓存
            calc = 0.0;
        }

        // 等待更新的隐藏层位置游标
        int hideAndOutputWeightLayerPoint = 0;
        // 当前隐藏层
        double hiddenNum = 0.0;
        // 暂存计算结果
        calc = 0.0;
        for (int outputPoint = 0; outputPoint < OUTPUT_LAYER_SIZE; outputPoint++) {
            for (int hiddenPoint = 0; hiddenPoint < HIDDEN_LAYER_SIZE; hiddenPoint++) {
                hiddenNum = hideLayer[hiddenPoint];
                // 这里计算要更新的层位置，比如0-100，每层10个房间，第0层的十位数就是0，加上个位数inputPoint就是对应房间号码
                hideAndOutputWeightLayerPoint = (outputPoint * HIDDEN_LAYER_SIZE) + hiddenPoint;
                calc = calc + (hiddenNum * hideAndOutputWeightLayer[hideAndOutputWeightLayerPoint]);
            }
            // 添加偏导数（用于平移函数图像）
            calc = calc + outputLayerBiasArray[outputPoint];
            // 计算完成，更新到outputLayer中
            outputLayer[outputPoint] = calc;
            // 重置计算缓存
            calc = 0.0;
        }

        double[] softmax = softmax2();

        return softmax;
    }

    /**
     * 激活函数，其原理是
     * e^x的导数等于其自身，该函数形成一个类似S形状的函数图
     * 用于计算inputNumber的置信度
     * System.out.println(sigmoid(-6.9001));
     * System.out.println(sigmoid(-1));
     * System.out.println(sigmoid(-0.7));
     * System.out.println(sigmoid(-0.5));
     * System.out.println(sigmoid(-0.1));
     * System.out.println(sigmoid(0));
     * System.out.println(sigmoid(0.1));
     * System.out.println(sigmoid(0.3));
     * System.out.println(sigmoid(0.5));
     * System.out.println(sigmoid(0.7));
     * System.out.println(sigmoid(1));
     * System.out.println(sigmoid(5));
     * System.out.println(sigmoid(15));
     * System.out.println(sigmoid(27.5));
     * System.out.println(sigmoid(1528));
     *
     * 输出：
     * 0.0010066702493808706
     * 0.2689414213699951
     * 0.33181222783183395
     * 0.37754066879814546
     * 0.47502081252106
     * 0.5
     * 0.52497918747894
     * 0.574442516811659
     * 0.6224593312018546
     * 0.6681877721681662
     * 0.7310585786300049
     * 0.9933071490757153
     * 0.999999694097773
     * 0.99999999999886
     * 1.0
     *
     *
     * @param inputNumber
     * @return
     */
    private double sigmoid(double inputNumber) {
        if (inputNumber >= 0) {
            double t = Math.exp(-inputNumber);
            return 1 / (1 + t);
        } else {
            double t = Math.exp(inputNumber);
            return t / (1 + t);
        }
    }

    /**
     * 根据当前神经网络，计算出最终的结果，用于显示
     *
     * // 步骤1：计算指数
     * double exp0 = Math.exp(2.0);   // ≈ 7.389
     * double exp1 = Math.exp(1.0);   // ≈ 2.718
     * double exp2 = Math.exp(0.1);   // ≈ 1.105
     *
     * // 步骤2：计算指数和
     * double sumExp = exp0 + exp1 + exp2;  // ≈ 7.389 + 2.718 + 1.105 = 11.212
     *
     * // 步骤3：计算每个类别的概率
     * double p0 = exp0 / sumExp;  // ≈ 7.389 / 11.212 = 0.659
     * double p1 = exp1 / sumExp;  // ≈ 2.718 / 11.212 = 0.242
     * double p2 = exp2 / sumExp;  // ≈ 1.105 / 11.212 = 0.099
     *
     * // 验证：总和应为1（近似）
     * System.out.println(p0 + p1 + p2);  // 输出：1.000
     */
    private double[] softmax() {
        double[] softmaxArray = new double[OUTPUT_LAYER_SIZE];
        double sum = 0;
        // 计算分母
        for (int point = 0; point < OUTPUT_LAYER_SIZE; point++) {
            softmaxArray[point] = Math.exp(outputLayer[point]);
            sum += softmaxArray[point];
        }
        // 计算softmax，并将计算结果赋值到输出层
        for (int point = 0; point < OUTPUT_LAYER_SIZE; point++) {
            softmaxArray[point] = softmaxArray[point] / sum;
        }
        return softmaxArray;
    }

    private double[] softmax2() {
        double[] softmaxArray = new double[OUTPUT_LAYER_SIZE];
        // 找最大值（防溢出）
        double max = outputLayer[0];
        for (int i = 1; i < OUTPUT_LAYER_SIZE; i++) {
            if (outputLayer[i] > max) {
                max = outputLayer[i];
            }
        }
        // exp(z_i - max)
        double sum = 0.0;
        for (int i = 0; i < OUTPUT_LAYER_SIZE; i++) {
            softmaxArray[i] = Math.exp(outputLayer[i] - max);
            sum += softmaxArray[i];
        }
        // 归一化
        for (int i = 0; i < OUTPUT_LAYER_SIZE; i++) {
            softmaxArray[i] /= sum;
        }
        return softmaxArray;
    }

    /**
     * 使用答案构造向量，用于反向传播
     *
     * @return
     */
    private double[] answerConstructionVector() {
        double[] vector = new double[OUTPUT_LAYER_SIZE];
        vector[answer] = 1;
        return vector;
    }

    /**
     * 打印出来所有的权重数据和偏置数
     */
    public void printAllData() {
        System.out.println("=======================================");
        System.out.println("输入层大小：" + this.INPUT_LAYER_SIZE);
        System.out.println("隐藏层大小：" + this.HIDDEN_LAYER_SIZE);
        System.out.println("输出层大小：" + this.OUTPUT_LAYER_SIZE);
        System.out.println("学习率：" + this.LEARNING_RATE);
        System.out.println("权重1：" + Arrays.toString(this.inputAndHideWeightLayer));
        System.out.println("偏置1：" + Arrays.toString(this.hiddenLayerBiasArray));
        System.out.println("权重2：" + Arrays.toString(this.hideAndOutputWeightLayer));
        System.out.println("偏置2：" + Arrays.toString(this.outputLayerBiasArray));
    }


}
