package ai.skymind;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;

public class Mnist_FNN {
    final static int seed = 1234;
    final static int batchSize = 500;
    final static int epoch = 5;

    public static void main(String[] args) throws IOException {
        //initialize iterator
        MnistDataSetIterator trainMnist = new MnistDataSetIterator(batchSize, true, seed);
        MnistDataSetIterator testMnist = new MnistDataSetIterator(batchSize, false, seed);

        //preprocessing
        NormalizerMinMaxScaler scaler = new NormalizerMinMaxScaler(0, 1);
        scaler.fit(trainMnist);  //fit the scaler to training data
        //apply for both datasets
        trainMnist.setPreProcessor(scaler);
        testMnist.setPreProcessor(scaler);

        //setting model config
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam())  //optimizer
                .weightInit(WeightInit.XAVIER)  //weight initialization
                .activation(Activation.RELU)  //default activation function
                .list()
                .layer(new DenseLayer.Builder()  //input layer
                        .nIn(trainMnist.inputColumns())  //specify the no. of examples in training data
                        .nOut(124)
                        .build())
                .layer(new DenseLayer.Builder()  //hidden layer 1
                        .nOut(282)
                        .build())
                .layer(new OutputLayer.Builder()  //output layer
                        .lossFunction(LossFunctions.LossFunction.MCXENT)  //multiclass cross-entropy loss
                        .activation(Activation.SOFTMAX)
                        .nOut(10)  //specifiy the class nunmber for datasets
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config); //construct model
        model.init();  //initialize model

        //set up server to record training statistics
        InMemoryStatsStorage storage = new InMemoryStatsStorage(); //allocate some memory
        UIServer server = UIServer.getInstance();
        server.attach(storage);  //attach memory to use for server
        model.setListeners(new StatsListener(storage), new ScoreIterationListener(1000));  //print per 1000 iteration

        //training for set epoch
        for (int i=0; i<epoch; i++)
        {
            model.fit(trainMnist);
        }

        //evaluation
        Evaluation evalTrain = model.evaluate(trainMnist);  //validation with training data
        Evaluation evalTest = model.evaluate(testMnist);  //test with testing data

        System.out.println("Training accuracy: " + evalTrain.accuracy());
        System.out.println("Testing  accuracy: " + evalTest.accuracy());


        System.out.println(evalTest.confusionMatrix());
    }
}
