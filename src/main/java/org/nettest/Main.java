package org.nettest;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

public class Main {

    private static final SimpleNet simpleNet = new SimpleNet();

    public static void main(String[] args) {

        // 训练10轮
        train(150);

        // 测试
        // 1
        int[] test01 = {0, 0, 0, 0, 0, 1, 0, 0, 0};
        // 2
        int[] test02 = {0, 0, 0, 0, 0, 0, 0, 0, 1};
        // 2
        int[] test03 = {0, 0, 0, 0, 0, 0, 0, 1, 1};
        // 3
        int[] test04 = {1, 0, 0, 0, 0, 0, 1, 0, 0};
        // 0
        int[] test05 = {0, 1, 1, 0, 0, 0, 0, 0, 0};
        // 0
        int[] test06 = {1, 0, 0, 0, 0, 0, 0, 0, 0};
        // 3
        int[] test07 = {1, 0, 0, 0, 0, 0, 0, 0, 0};
        // 3
        int[] test08 = {1, 0, 0, 1, 0, 0, 1, 0, 0};

        test(1, test01);
        test(2, test02);
        test(2, test03);
        test(3, test04);
        test(0, test05);
        test(0, test06);
        test(3, test07);
        test(3, test08);

        // 输出训练结果
        simpleNet.printAllData();
    }

    /**
     * 测试
     */
//    public static void test(int answer, int... arr) {
//        double[] predict = simpleNet.predict(arr);
//        double maxValue = 0.0;
//        int maxIndex = 0;
//        String[] pointArr = {"↑", "→", "↓", "←"};
//        for (int i = 0; i < predict.length; i++) {
//            if (predict[i] >= maxValue) {
//                maxValue = predict[i];
//                maxIndex = i;
//            }
//        }
//        System.out.println("输入：");
//        for (int i = 0; i < 3; i++) {
//            for (int j = 0; j < 3; j++) {
//                if (i * 3 + j == 4) {
//                    System.out.print(pointArr[maxIndex]);
//                    continue;
//                }
//                String str = arr[i * 3 + j] == 1 ? "■" : "□";
//                System.out.print(str);
//            }
//            System.out.println();
//        }
//        System.out.println("结果：" + Arrays.toString(predict));
//        System.out.println("答案：1");
//    }

    public static void test(int answer, int... arr) {
        double[] predict = simpleNet.predict(arr);
        double maxValue = 0.0;
        int maxIndex = 0;
        String[] pointArr = {"↑", "→", "↓", "←"};

        for (int i = 0; i < predict.length; i++) {
            if (predict[i] >= maxValue) {
                maxValue = predict[i];
                maxIndex = i;
            }
        }

        System.out.println("\n测试输入（答案：" + answer + "，预测：" + maxIndex + "）：");
        // 显示输入
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                int index = i * 3 + j;
                if (index == 4) {
                    System.out.print(pointArr[maxIndex] + " ");
                    continue;
                }
                String str = arr[index] == 1 ? "■" : "□";
                System.out.print(str + " ");
            }
            System.out.println();
        }
        System.out.printf("预测概率：↑%.2f →%.2f ↓%.2f ←%.2f\n",
                predict[0], predict[1], predict[2], predict[3]);
        System.out.println("是否正确：" + (maxIndex == answer ? "✓" : "✗"));
    }

    /**
     * 训练
     *
     * @param count 轮数
     */
    public static void train(int count) {
        count = count <= 1 ? 1 : count;
        // 初始化权重
        simpleNet.initWeightLayer();
        for (int i = 0; i < count; i++) {
            System.out.println("第" + (i + 1) + "轮");
            dataMap.forEach((answer, list) -> {
                for (Integer[] vector : list) {
                    int[] array = Arrays.stream(vector)
                            .mapToInt(Integer::intValue)
                            .toArray();
                    simpleNet.training(answer, array);
                }
            });
            System.out.println("第" + (i + 1) + "轮结束");
        }

    }

    public static void testSigmoid() {
        System.out.println(sigmoid(-6.9001));
        System.out.println(sigmoid(-1));
        System.out.println(sigmoid(-0.7));
        System.out.println(sigmoid(-0.5));
        System.out.println(sigmoid(-0.1));
        System.out.println(sigmoid(0));
        System.out.println(sigmoid(0.1));
        System.out.println(sigmoid(0.3));
        System.out.println(sigmoid(0.5));
        System.out.println(sigmoid(0.7));
        System.out.println(sigmoid(1));
        System.out.println(sigmoid(5));
        System.out.println(sigmoid(15));
        System.out.println(sigmoid(27.5));
        System.out.println(sigmoid(1528));
    }

    /**
     * 激活函数，其原理是
     * e^x的导数等于其自身，该函数形成一个类似S形状的函数图
     * 用于计算inputNumber的确定值
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
     * <p>
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
     * @param inputNumber
     * @return
     */
    private static double sigmoid(double inputNumber) {
        if (inputNumber >= 0) {
            double t = Math.exp(-inputNumber);
            return 1 / (1 + t);
        } else {
            double t = Math.exp(inputNumber);
            return t / (1 + t);
        }
    }


    /**
     * 定义数据：
     * 上=0，右=1，下=2，左=3
     * 数据格子设置如下：
     * 0 1 2
     * 3 4 5
     * 6 7 8
     */
    public static HashMap<Integer, List<Integer[]>> dataMap = new HashMap<>() {
        {
            put(0, Arrays.asList(
                    new Integer[]{0, 1, 0, 0, 0, 0, 0, 0, 0},
                    new Integer[]{1, 0, 1, 0, 0, 0, 0, 0, 0},
                    new Integer[]{1, 0, 0, 0, 0, 0, 0, 0, 0},
                    new Integer[]{0, 0, 1, 0, 0, 0, 0, 0, 0},
                    new Integer[]{1, 1, 0, 0, 0, 0, 0, 0, 0},
                    new Integer[]{0, 1, 1, 0, 0, 0, 0, 0, 0},
                    new Integer[]{1, 1, 1, 0, 0, 0, 0, 0, 0})
            );
            put(1, Arrays.asList(
                    new Integer[]{0, 0, 1, 0, 0, 0, 0, 0, 0},
                    new Integer[]{0, 0, 0, 0, 0, 1, 0, 0, 0},
                    new Integer[]{0, 0, 0, 0, 0, 0, 0, 0, 1},
                    new Integer[]{0, 0, 1, 0, 0, 0, 0, 0, 1},
                    new Integer[]{0, 0, 1, 0, 0, 1, 0, 0, 0},
                    new Integer[]{0, 0, 0, 0, 0, 1, 0, 0, 1},
                    new Integer[]{0, 0, 1, 0, 0, 1, 0, 0, 1}
            ));

            put(2, Arrays.asList(
                    new Integer[]{0, 0, 0, 0, 0, 0, 1, 0, 0},
                    new Integer[]{0, 0, 0, 0, 0, 0, 0, 1, 0},
                    new Integer[]{0, 0, 0, 0, 0, 0, 0, 0, 1},
                    new Integer[]{0, 0, 0, 0, 0, 0, 1, 1, 0},
                    new Integer[]{0, 0, 0, 0, 0, 0, 0, 1, 1},
                    new Integer[]{0, 0, 0, 0, 0, 0, 1, 0, 1},
                    new Integer[]{0, 0, 0, 0, 0, 0, 1, 1, 1}
            ));

            put(3, Arrays.asList(
                    new Integer[]{1, 0, 0, 0, 0, 0, 0, 0, 0},
                    new Integer[]{0, 0, 0, 1, 0, 0, 0, 0, 0},
                    new Integer[]{0, 0, 0, 0, 0, 0, 1, 0, 0},
                    new Integer[]{1, 0, 0, 1, 0, 0, 0, 0, 0},
                    new Integer[]{0, 0, 0, 1, 0, 0, 1, 0, 0},
                    new Integer[]{1, 0, 0, 0, 0, 0, 1, 0, 0},
                    new Integer[]{1, 0, 0, 1, 0, 0, 1, 0, 0}
            ));
        }
    };
}

