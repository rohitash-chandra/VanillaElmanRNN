/*
 Dr. Rohitash Chandra.
 Coded in 2005. Updated 2016.
 Based in: Werbos, Paul J. "Backpropagation through time: what it does and how to do it." Proceedings of the IEEE 78.10 (1990): 1550-1560.
 
 https://en.wikipedia.org/wiki/Backpropagation_through_time
 
 based on: https://github.com/rohitash-chandra/feedforward-neural-network
 */
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <ctime>

#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<ctype.h>

time_t TicTime;
time_t TocTime;
using namespace ::std;

typedef vector<double> Layer;
typedef vector<double> Nodes;
typedef vector<double> Frame;
typedef vector<int> Sizes;
typedef vector<vector<double> > Weight;
typedef vector<vector<double> > Data;

const double MinimumError = 0.00001;

const int trainsize = 299;
const int testsize = 99;

char * trainfile = "train_embed.txt"; // time series after state state reconstruction (time lag = 2, dimen = 3)
char * testfile = "test_embed.txt"; //

#define sigmoid //tanh, sigmoid what to use in output layer

const int maxtime = 1000; // max time in epocs

const double learningRate = 0.2;

const double weightdecay = 0.01;

const double MAEscalingfactor = 10;

const int LayersNumber = 3; //total number of layers.

const int MaxVirtLayerSize = 50; // max unfolds in time

int row;
int col;
int layer;
int r;
int x;
int y;

int RUN;

class Samples {
public:

	Data InputValues;
	Data DataSet;
	Layer OutputValues;
	int PhoneSize;
	//int SampleSize;
public:
	Samples() {
	}

};

typedef vector<Samples> DataSample;

class TrainingExamples {
public:

	char* FileName;

	int SampleSize;
	int ColumnSize;
	int OutputSize;
	int RowSize;

	DataSample Sample;

	// Samples Sample[MaxSampSize];

public:
	TrainingExamples() {

	}
	;

	TrainingExamples(char* File, int sampleSize, int columnSize,
			int outputSize) {

		Samples sample;

		for (int i = 0; i < sampleSize; i++) {
			Sample.push_back(sample);
		}

		int rows;
		RowSize = MaxVirtLayerSize; // max number of rows.
		ColumnSize = columnSize;
		SampleSize = sampleSize;
		OutputSize = outputSize;

		ifstream in(File);

		//initialise input vectors
		for (int sample = 0; sample < SampleSize; sample++) {

			for (int r = 0; r < RowSize; r++)
				Sample[sample].InputValues.push_back(vector<double>());

			for (int row = 0; row < RowSize; row++) {
				for (int col = 0; col < ColumnSize; col++)
					Sample[sample].InputValues[row].push_back(0);
			}

			for (int out = 0; out < OutputSize; out++)
				Sample[sample].OutputValues.push_back(0);
		}
		//---------------------------------------------

		for (int samp = 0; samp < SampleSize; samp++) {
			in >> rows;
			Sample[samp].PhoneSize = rows;

			for (row = 0; row < Sample[samp].PhoneSize; row++) {
				for (col = 0; col < ColumnSize; col++)
					in >> Sample[samp].InputValues[row][col];
			}

			for (int out = 0; out < OutputSize; out++)
				in >> Sample[samp].OutputValues[out];

			// cout<<rows<<endl;
		}

		cout << "printing..." << endl;

		in.close();
	}

	void printData();

};
//.................................................

void TrainingExamples::printData() {
	for (int sample = 0; sample < SampleSize; sample++) {
		for (row = 0; row < Sample[sample].PhoneSize; row++) {
			for (col = 0; col < ColumnSize; col++)
				cout << Sample[sample].InputValues[row][col] << " ";
			cout << endl;
		}
		cout << endl;
		for (int out = 0; out < OutputSize; out++)
			cout << " " << Sample[sample].OutputValues[out] << " ";

		cout << endl << "--------------" << endl;
	}
}

//*********************************************************
class Layers {
public:

	double Weights[35][35];
	double WeightChange[35][35];
	double ContextWeight[35][35];

	Weight TransitionProb;

	Data RadialOutput;
	Data Outputlayer;
	Layer Bias;
	Layer BiasChange;
	Data Error;

	Layer Mean;
	Layer StanDev;

	Layer MeanChange;
	Layer StanDevChange;

public:
	Layers() {

	}

};

//***************************************************

class NeuralNetwork: public virtual TrainingExamples {
public:

	Layers nLayer[LayersNumber];

	double Heuristic;
	Layer ChromeNeuron;
	Data Output;
	double NMSE;
	int StringSize;

	Sizes layersize;
public:

	NeuralNetwork(Sizes layer) {
		layersize = layer;

		StringSize = (layer[0] * layer[1]) + (layer[1] * layer[2])
				+ (layer[1] * layer[1]) + (layer[1] + layer[2]);

	}
	NeuralNetwork() {
	}

	double Random();

	double Sigmoid(double ForwardOutput);
	double SigmoidS(double ForwardOutput);
	double NMSError() {
		return NMSE;
	}

	void CreateNetwork(Sizes Layersize, int Maxsize);

	void ForwardPass(Samples Sample, int patternNum, Sizes Layersize,
			int phone);

	void BackwardPass(Samples Sample, double LearningRate, int slide,
			Sizes Layersize, int phone);

	void PrintWeights(Sizes Layersize); // print  all weights
	//
	bool ErrorTolerance(TrainingExamples TraineeSamples, int temp,
			Sizes Layersize);

	double SumSquaredError(TrainingExamples TraineeSamples, int temp,
			Sizes Layersize);

	int BackPropogation(TrainingExamples TraineeSamples, double LearningRate,
			Sizes Layersize, char* Savefile, char* TestFile, int sampleSize,
			int columnSize, int outputSize, ofstream &out1, ofstream &out2);

	void SaveLearnedData(Sizes Layersize, char* filename);

	void LoadSavedData(Sizes Layersize, char* filename);

	double TestLearnedData(Sizes Layersize, char* learntData, char* TestFile,
			int sampleSize, int columnSize, int outputSize);

	double CountLearningData(TrainingExamples TraineeSamples, int temp,
			Sizes Layersize);

	double CountTestingData(TrainingExamples TraineeSamples, int temp,
			Sizes Layersize);

	double MAE(TrainingExamples TraineeSamples, int temp, Sizes Layersize);

	void ChoromesToNeurons(Layer NeuronChrome);

	double ForwardFitnessPass(Layer NeuronChrome, TrainingExamples Test);

	bool CheckOutput(TrainingExamples TraineeSamples, int pattern,
			Sizes Layersize);

	double TestTrainingData(Sizes Layersize, char* learntData, char* TestFile,
			int sampleSize, int columnSize, int outputSize, ofstream & out2);

	double NormalisedMeanSquaredError(TrainingExamples TraineeSamples, int temp,
			Sizes Layersize);

	double BP(Layer NeuronChrome, TrainingExamples Test, int generations);

	double Abs(double num);

	double MAPE(TrainingExamples TraineeSamples, int temp, Sizes Layersize);

};

double NeuralNetwork::Random() {
	int chance;
	double randomWeight = 0;
	double NegativeWeight = 0;
	chance = rand() % 2;

	if (chance == 0) {

		return drand48() * 0.5;
	}

	if (chance == 1) {

		return drand48() * 0.5;
	}

}
double NeuralNetwork::Sigmoid(double ForwardOutput) {
	double ActualOutput;
#ifdef sigmoid
	ActualOutput = (1.0 / (1.0 + exp(-1.0 * (ForwardOutput))));
#endif

#ifdef tanh
	ActualOutput = (exp(2 * ForwardOutput) - 1)/(exp(2 * ForwardOutput) + 1);
#endif
	return ActualOutput;
}

double NeuralNetwork::SigmoidS(double ForwardOutput) {
	double ActualOutput;

	ActualOutput = (1.0 / (1.0 + exp(-1.0 * (ForwardOutput))));

	return ActualOutput;
}

double NeuralNetwork::Abs(double num) {
	if (num < 0)
		return num * -1;
	else
		return num;
}

void NeuralNetwork::CreateNetwork(Sizes Layersize, int Maxsize) {

	int end = Layersize.size() - 1;

	for (layer = 0; layer < Layersize.size() - 1; layer++) {

		//-------------------------------------------
		//for( r=0; r < Layersize[layer]; r++)
		//nLayer[layer].Weights.push_back(vector<double> ());

		for (row = 0; row < Layersize[layer]; row++)
			for (col = 0; col < Layersize[layer + 1]; col++)
				nLayer[layer].Weights[row][col] = Random();
		//---------------------------------------------

		for (row = 0; row < Layersize[layer]; row++)
			for (col = 0; col < Layersize[layer + 1]; col++)
				nLayer[layer].WeightChange[row][col] = Random();
		//-------------------------------------------
	}

	for (row = 0; row < Layersize[1]; row++)
		for (col = 0; col < Layersize[1]; col++)
			nLayer[1].ContextWeight[row][col] = Random();

	//}
	//------------------------------------------------------

	for (layer = 0; layer < Layersize.size(); layer++) {

		for (r = 0; r < Maxsize; r++)
			nLayer[layer].Outputlayer.push_back(vector<double>());

		for (row = 0; row < Maxsize; row++)
			for (col = 0; col < Layersize[layer]; col++)
				nLayer[layer].Outputlayer[row].push_back(Random());
		//--------------------------------------------------

		for (r = 0; r < MaxVirtLayerSize; r++)
			nLayer[layer].Error.push_back(vector<double>());

		for (row = 0; row < MaxVirtLayerSize; row++)
			for (col = 0; col < Layersize[layer]; col++)
				nLayer[layer].Error[row].push_back(0);

		//TransitionProb
		//---------------------------------------------
		for (r = 0; r < MaxVirtLayerSize; r++)
			nLayer[layer].RadialOutput.push_back(vector<double>());

		for (row = 0; row < MaxVirtLayerSize; row++)
			for (col = 0; col < Layersize[layer]; col++)
				nLayer[layer].RadialOutput[row].push_back(Random());
		//-------------------------------------------

		//---------------------------------------------

		for (row = 0; row < Layersize[layer]; row++)
			nLayer[layer].Bias.push_back(Random());

		for (row = 0; row < Layersize[layer]; row++)
			nLayer[layer].BiasChange.push_back(0);

		for (row = 0; row < Layersize[layer]; row++)
			nLayer[layer].Mean.push_back(Random());

		for (row = 0; row < Layersize[layer]; row++)
			nLayer[layer].StanDev.push_back(Random());

		for (row = 0; row < Layersize[layer]; row++)
			nLayer[layer].MeanChange.push_back(0);

		for (row = 0; row < Layersize[layer]; row++)
			nLayer[layer].StanDevChange.push_back(0);

	}
	//--------------------------------------

	for (r = 0; r < Maxsize; r++)
		Output.push_back(vector<double>());
	for (row = 0; row < Maxsize; row++)
		for (col = 0; col < Layersize[end]; col++)
			Output[row].push_back(0);

	for (row = 0; row < StringSize; row++)
		ChromeNeuron.push_back(0);

	// SaveLearnedData(Layersize, "createnetwork.txt");

}

void NeuralNetwork::ForwardPass(Samples Sample, int slide, Sizes Layersize,
		int phone) {
	double WeightedSum = 0;
	double ContextWeightSum = 0;
	double ForwardOutput;
	//  cout<<endl<<"slide  "<<slide<<"  ------------------------ "<<endl<<endl<<endl<<endl;

	int end = Layersize.size() - 1;

	for (int row = 0; row < Layersize[0]; row++)
		nLayer[0].Outputlayer[slide + 1][row] = Sample.InputValues[slide][row];
	//--------------------------------------------

	//for(
	int layer = 0; // layer < Layersize.size()-1; layer++){
	int y;
	int x;
	for (y = 0; y < Layersize[layer + 1]; y++) {
		for (x = 0; x < Layersize[layer]; x++) {
			WeightedSum += (nLayer[layer].Outputlayer[slide + 1][x]
					* nLayer[layer].Weights[x][y]);
		}
		for (x = 0; x < Layersize[layer + 1]; x++) {
			ContextWeightSum += (nLayer[1].Outputlayer[slide][x]
					* nLayer[1].ContextWeight[x][y]); // adjust this line when use two hidden layers.
			//
		}

		ForwardOutput = (WeightedSum + ContextWeightSum)
				- nLayer[layer + 1].Bias[y];
		nLayer[layer + 1].Outputlayer[slide + 1][y] = SigmoidS(ForwardOutput);
		// cout<<ForwardOutput<<endl;
		//getchar();
		WeightedSum = 0;
		ContextWeightSum = 0;
	}
//}//end layer

	layer = 1;

	for (y = 0; y < Layersize[layer + 1]; y++) {
		for (x = 0; x < Layersize[layer]; x++) {
			WeightedSum += (nLayer[layer].Outputlayer[slide + 1][x]
					* nLayer[layer].Weights[x][y]);
			ForwardOutput = (WeightedSum) - nLayer[layer + 1].Bias[y];
		}
		nLayer[layer + 1].Outputlayer[slide + 1][y] = Sigmoid(ForwardOutput);
		WeightedSum = 0;
		//cout<<   ForwardOutput<<endl;
		ContextWeightSum = 0;
	}

	//--------------------------------------------
	for (int output = 0; output < Layersize[end]; output++) {
		Output[phone][output] = nLayer[end].Outputlayer[slide + 1][output];
//cout<<Output[phone][output]<<" ";
	}
//cout<<endl;

}

void NeuralNetwork::BackwardPass(Samples Sample, double LearningRate, int slide,
		Sizes Layersize, int phone) {

	int end = Layersize.size() - 1; // know the end layer
	double temp = 0;
	double sum = 0;
	int Endslide = Sample.PhoneSize;
	//----------------------------------------
//cout<<slide<<"   ---------------------------------->>>>"<<endl;
	// compute error gradient for output neurons
	for (int output = 0; output < Layersize[end]; output++) {
		nLayer[2].Error[Endslide][output] = 1
				* (Sample.OutputValues[output] - Output[phone][output]);

	}
//

	//----------------------------------------
	// for(int layer = Layersize.size()-2; layer >= 0; layer--){
	int layer = 1;
	for (x = 0; x < Layersize[layer]; x++) { //inner layer
		for (y = 0; y < Layersize[layer + 1]; y++) { //outer layer
			temp += (nLayer[layer + 1].Error[Endslide][y]
					* nLayer[layer].Weights[x][y]);
		}

		nLayer[layer].Error[Endslide][x] =
				nLayer[layer].Outputlayer[Endslide][x]
						* (1 - nLayer[layer].Outputlayer[Endslide][x]) * temp;
		temp = 0.0;
	}
	// }
//cout<<nLayer[1].Error[Endslide][0]<<"        eeee"<<endl;
	for (x = 0; x < Layersize[1]; x++) { //inner layer
		for (y = 0; y < Layersize[1]; y++) { //outer layer
			sum += (nLayer[1].Error[slide][y] * nLayer[1].ContextWeight[x][y]);
//cout<<sum<< " is sum  : "<<nLayer[1].Error[slide][y]<<endl;

		}
		nLayer[1].Error[slide - 1][x] = (nLayer[1].Outputlayer[slide - 1][x]
				* (1 - nLayer[1].Outputlayer[slide - 1][x])) * sum;
		//	cout<<	nLayer[1].Error[slide-1][x]<<" is error  of slide "<<slide<<endl;
		sum = 0.0;
	}
	sum = 0.0;

// do weight updates..
//---------------------------------------
	double tmp;
//for( layer = Layersize.size()-2; layer != -1; layer--){
	for (x = 0; x < Layersize[0]; x++) { //inner layer
		for (y = 0; y < Layersize[1]; y++) { //outer layer
			tmp = ((LearningRate * nLayer[1].Error[slide][y]
					* nLayer[0].Outputlayer[slide][x])); // weight change
			nLayer[0].Weights[x][y] += tmp - (tmp * weightdecay);
		}

	}
	// }

//cout<<endl;

//-------------------------------------------------
//do top weight update
	double seeda = 0;

	//if(Endslide ==  slide)
	seeda = 1;
	double tmpoo;
//for( layer = Layersize.size()-2; layer != -1; layer--){
	for (x = 0; x < Layersize[1]; x++) { //inner layer
		for (y = 0; y < Layersize[2]; y++) { //outer layer
			tmpoo = ((seeda * LearningRate * nLayer[2].Error[Endslide][y]
					* nLayer[1].Outputlayer[Endslide][x])); // weight change
			nLayer[1].Weights[x][y] += tmpoo - (tmpoo * weightdecay);

		}

	}
	seeda = 0;
	// }

	//-----------------------------------------------
	double tmp2;
//for( layer = Layersize.size()-2; layer != -1; layer--){
	for (x = 0; x < Layersize[1]; x++) { //inner layer
		for (y = 0; y < Layersize[1]; y++) { //outer layer
			tmp2 = ((LearningRate * nLayer[1].Error[slide][y]
					* nLayer[1].Outputlayer[slide - 1][x])); // weight change
			nLayer[1].ContextWeight[x][y] += tmp2 - (tmp2 * weightdecay);

		}

	}

//update the bias
	double topbias = 0;
	double seed = 0;

	//if(Endslide ==  slide)
	seed = 1;
	for (y = 0; y < Layersize[2]; y++) {
		topbias = ((seed * -1 * LearningRate * nLayer[2].Error[Endslide][y]));
		nLayer[2].Bias[y] += topbias - (topbias * weightdecay);
		topbias = 0;
		//   	cout<<nLayer[2].Bias[y]<<" is updated  top Bias for slide "<<Endslide<<endl;
	}
	topbias = 0;
	seed = 0;

	double tmp1;

	for (y = 0; y < Layersize[1]; y++) {
		tmp1 = ((-1 * LearningRate * nLayer[1].Error[slide][y]));
		nLayer[1].Bias[y] += tmp1 - (tmp1 * weightdecay);

	}
	// }

}

double NeuralNetwork::MAPE(TrainingExamples TraineeSamples, int temp,
		Sizes Layersize) {
	int end = Layersize.size() - 1;
	double Sum = 0;
	double Error = 0;
	double ErrorSquared = 0;
	for (int pattern = 0; pattern < temp; pattern++) {
		for (int output = 0; output < Layersize[end]; output++) {
			Error = (TraineeSamples.Sample[pattern].OutputValues[output]
					- Output[pattern][output])
					/ TraineeSamples.Sample[pattern].OutputValues[output];

			ErrorSquared += fabs(Error);
		}

		Sum += (ErrorSquared);
		ErrorSquared = 0;
	}
	return (Sum / temp * Layersize[end] * 100);

}

double NeuralNetwork::MAE(TrainingExamples TraineeSamples, int temp,
		Sizes Layersize) {
	int end = Layersize.size() - 1;
	double Sum = 0;
	double Error = 0;
	double ErrorSquared = 0;
	for (int pattern = 0; pattern < temp; pattern++) {
		for (int output = 0; output < Layersize[end]; output++) {
			Error = (TraineeSamples.Sample[pattern].OutputValues[output]
					- Output[pattern][output]) * MAEscalingfactor;

			ErrorSquared += fabs(Error);
		}

		Sum += (ErrorSquared);
		ErrorSquared = 0;
	}
	return Sum / temp * Layersize[end];

}
double NeuralNetwork::SumSquaredError(TrainingExamples TraineeSamples, int temp,
		Sizes Layersize) {
	int end = Layersize.size() - 1;
	double Sum = 0;
	double Error = 0;
	double ErrorSquared = 0;
	for (int pattern = 0; pattern < temp; pattern++) {
		for (int output = 0; output < Layersize[end]; output++) {
			Error = TraineeSamples.Sample[pattern].OutputValues[output]
					- Output[pattern][output];

			ErrorSquared += (Error * Error);
		}

		Sum += (ErrorSquared);
		ErrorSquared = 0;
	}
	return sqrt(Sum / temp * Layersize[end]);

//return MAPE(TraineeSamples,temp,Layersize);
}

double NeuralNetwork::NormalisedMeanSquaredError(
		TrainingExamples TraineeSamples, int temp, Sizes Layersize) {
	int end = Layersize.size() - 1;
	double Sum = 0;
	double Sum2 = 0;
	double Error = 0;
	double ErrorSquared = 0;
	double Error2 = 0;
	double ErrorSquared2 = 0;
	double meany = 0;
	for (int pattern = 0; pattern < temp; pattern++) {

		for (int slide = 0; slide < TraineeSamples.Sample[pattern].PhoneSize;
				slide++) {
			for (int input = 0; input < Layersize[0]; input++) {
				meany +=
						TraineeSamples.Sample[pattern].InputValues[slide][input];
			}
			meany /= Layersize[0] * TraineeSamples.Sample[pattern].PhoneSize;
		}

		for (int output = 0; output < Layersize[end]; output++) {
			Error2 = TraineeSamples.Sample[pattern].OutputValues[output]
					- meany;
			Error = TraineeSamples.Sample[pattern].OutputValues[output]
					- Output[pattern][output];
			ErrorSquared += (Error * Error);
			ErrorSquared2 += (Error2 * Error2);

		}
		meany = 0;
		Sum += (ErrorSquared);
		Sum2 += (ErrorSquared2);
		ErrorSquared = 0;
		ErrorSquared2 = 0;
	}

	return Sum / Sum2;
}

void NeuralNetwork::PrintWeights(Sizes Layersize) {
	int end = Layersize.size() - 1;

	for (int layer = 0; layer < Layersize.size() - 1; layer++) {

		cout << layer << "  Weights::" << endl << endl;
		for (int row = 0; row < Layersize[layer]; row++) {
			for (int col = 0; col < Layersize[layer + 1]; col++)
				cout << nLayer[layer].Weights[row][col] << " ";
			cout << endl;
		}
		cout << endl << layer << " ContextWeight::" << endl << endl;

		for (int row = 0; row < Layersize[1]; row++) {
			for (int col = 0; col < Layersize[1]; col++)
				cout << nLayer[1].ContextWeight[row][col] << " ";
			cout << endl;
		}

	}

}
//-------------------------------------------------------

void NeuralNetwork::SaveLearnedData(Sizes Layersize, char* filename) {

	ofstream out;
	out.open(filename);
	if (!out) {
		cout << endl << "failed to save file" << endl;
		return;
	}
	//-------------------------------
	for (int layer = 0; layer < Layersize.size() - 1; layer++) {
		for (int row = 0; row < Layersize[layer]; row++) {
			for (int col = 0; col < Layersize[layer + 1]; col++)
				out << nLayer[layer].Weights[row][col] << " ";
			out << endl;
		}
		out << endl;
	}
	//-------------------------------
	for (int row = 0; row < Layersize[1]; row++) {
		for (int col = 0; col < Layersize[1]; col++)
			out << nLayer[1].ContextWeight[row][col] << " ";
		out << endl;
	}
	out << endl;
	//--------------------------------
	// output bias.
	for (int layer = 1; layer < Layersize.size(); layer++) {
		for (int y = 0; y < Layersize[layer]; y++) {
			out << nLayer[layer].Bias[y] << "  ";
			out << endl;
		}

	}
	out << endl;
	//------------------------------
	out.close();
//	cout << endl << "data saved" << endl;

	return;
}

void NeuralNetwork::LoadSavedData(Sizes Layersize, char* filename) {
	ifstream in(filename);
	if (!in) {
		cout << endl << "failed to load file" << endl;
		return;
	}
	//-------------------------------
	for (int layer = 0; layer < Layersize.size() - 1; layer++)
		for (int row = 0; row < Layersize[layer]; row++)
			for (int col = 0; col < Layersize[layer + 1]; col++)
				in >> nLayer[layer].Weights[row][col];
	//---------------------------------
	for (int row = 0; row < Layersize[1]; row++)
		for (int col = 0; col < Layersize[1]; col++)
			in >> nLayer[1].ContextWeight[row][col];
	//--------------------------------
	// output bias.
	for (int layer = 1; layer < Layersize.size(); layer++)
		for (int y = 0; y < Layersize[layer]; y++)
			in >> nLayer[layer].Bias[y];

	in.close();
	// cout << endl << "data loaded for testing" << endl;

	return;
}

double NeuralNetwork::TestTrainingData(Sizes Layersize, char* learntData,
		char* TestFile, int sampleSize, int columnSize, int outputSize,
		ofstream & out2) {
	bool valid;
	double count = 1;
	double total;
	double accuracy;
	int end = Layersize.size() - 1;
	Samples sample;
	TrainingExamples Test(TestFile, sampleSize, columnSize, outputSize);

	for (int phone = 0; phone < Test.SampleSize; phone++) {
		sample = Test.Sample[phone];

		int slide;

		for (slide = 0; slide < sample.PhoneSize; slide++) {
			ForwardPass(sample, slide, Layersize, phone);

		}
	}

	// for(int pattern = 0; pattern< Test.SampleSize; pattern++){     //in case if you wish to print the prediction outputs
	//out2<< Output[pattern][0]  <<" "<<Test.Sample[pattern].OutputValues[0] <<endl;

	//  }
//out2<<endl;

	out2 << endl;
	accuracy = SumSquaredError(Test, Test.SampleSize, Layersize);
	out2 << " RMSE:  " << accuracy << endl;
	cout << "RMSE: " << accuracy << " %" << endl;
	NMSE = MAE(Test, Test.SampleSize, Layersize);
	out2 << " NMSE:  " << NMSE << endl;
	return accuracy;
}

int NeuralNetwork::BackPropogation(TrainingExamples TraineeSamples,
		double LearningRate, Sizes Layersize, char * Savefile, char* TestFile,
		int sampleSize, int columnSize, int outputSize, ofstream &out1,
		ofstream &out2) {

	double SumErrorSquared;

	Sizes Array;

	Samples sample;

	CreateNetwork(Layersize, TraineeSamples.SampleSize);

	int Id = 0;

	int c = 1;

	for (int epoch = 0; epoch < maxtime; epoch++) {

		//TraineeSamples.SampleSize
		for (int phone = 0; phone < TraineeSamples.SampleSize; phone++) {
			sample = TraineeSamples.Sample[phone];

			int slide;

			for (slide = 0; slide < sample.PhoneSize; slide++) {

				ForwardPass(sample, slide, Layersize, phone);

			}

			for (slide = sample.PhoneSize; slide >= 1; slide--) {
				BackwardPass(sample, LearningRate, slide, Layersize, phone);
			}

		}

		double Train = 0;
		if (epoch % 100 == 0) {
			SumErrorSquared = SumSquaredError(TraineeSamples,
					TraineeSamples.SampleSize, Layersize);

			double mae = MAE(TraineeSamples, TraineeSamples.SampleSize,
					Layersize);
			cout << SumErrorSquared << "     " << mae << " " << epoch << endl;

			out1 << SumErrorSquared << "     " << mae << " " << " " << Train
					<< epoch << endl;
		}

	}

	SaveLearnedData(Layersize, Savefile);

	return c;

}
//*************************************************

class Simulation {

public:
	int TotalEval;
	int TotalSize;
	double Train;
	double Test;
	double TrainNMSE;
	double TestNMSE;
	double Error;

	int Cycles;
	bool Sucess;

	Simulation() {

	}

	int GetEval() {
		return TotalEval;
	}
	double GetCycle() {
		return Train;
	}
	double GetError() {
		return Test;
	}

	double NMSETrain() {
		return TrainNMSE;
	}
	double NMSETest() {
		return TestNMSE;
	}

	bool GetSucess() {
		return Sucess;
	}

	void Procedure(bool bp, double h, ofstream &out1, ofstream &out2,
			ofstream &out3);
};

void Simulation::Procedure(bool bp, double h, ofstream &out1, ofstream &out2,
		ofstream &out3) {

	clock_t start = clock();

	int hidden = h;

	int output = 1;
	int input = 1;

	int weightsize1 = (input * hidden);

	int weightsize2 = (hidden * output);

	int contextsize = hidden * hidden;
	int biasize = hidden + output;

	int gene = 1;
	double trainpercent = 0;
	double testpercent = 0;
	int epoch;
	double testtree;

	char file[15] = "Learnt.txt";
	TotalEval = 0;
	double H = 0;

	TrainingExamples Samples(trainfile, trainsize, input, output);
	// Samples.printData();

	double error;

	Sizes layersize;
	layersize.push_back(input);
	layersize.push_back(hidden);
	layersize.push_back(output);

	NeuralNetwork network(layersize);
	network.CreateNetwork(layersize, trainsize);

	epoch = network.BackPropogation(Samples, learningRate, layersize, file,
			trainfile, trainsize, input, output, out1, out2); //  train the network

	out2 << "Train" << endl;
	Train = network.TestTrainingData(layersize, file, trainfile, trainsize,
			input, output, out2);
	TrainNMSE = network.NMSError();
	out2 << "Test" << endl;
	Test = network.TestTrainingData(layersize, file, testfile, testsize, input,
			output, out2);
	TestNMSE = network.NMSError();
	out2 << endl;
	cout << Test << " was test RMSE " << endl;
	out1 << endl;
	out1 << " ------------------------------ " << h << "  " << TotalEval
			<< "  RMSE:  " << Train << "  " << Test << " NMSE:  " << TrainNMSE
			<< " " << TestNMSE << endl;

	out2 << " ------------------------------ " << h << "  " << TotalEval << "  "
			<< Train << "  " << Test << endl;
	out3 << "  " << h << "  " << TotalEval << "  RMSE:  " << Train << "  "
			<< Test << " NMSE:  " << TrainNMSE << " " << TestNMSE << endl;

}

//---------------------------------------------------------------------------------------
int main(void) {
	cout << "hello" << endl;

	int VSize = 90;

	ofstream out1;
	out1.open("Oneout1.txt");
	ofstream out2;
	out2.open("Oneout2.txt");
	ofstream out3;
	out3.open("Oneout3.txt");

	for (int hidden = 3; hidden <= 7; hidden += 2) {
		Sizes EvalAverage;
		Layer ErrorAverage;
		Layer CycleAverage;

		Layer NMSETrainAve;
		Layer NMSETestAve;

		int MeanEval = 0;
		double MeanError = 0;
		double MeanCycle = 0;

		double NMSETrainMean = 0;
		double NMSETestMean = 0;

		int EvalSum = 0;

		double NMSETrainSum = 0;
		double NMSETestSum = 0;

		double ErrorSum = 0;
		double CycleSum = 0;
		double maxrun = 30;
		int success = 0;

		double BestRMSE = 3;
		double BestNMSE = 3;

		for (int run = 1; run <= maxrun; run++) {
			Simulation Combined;

			Combined.Procedure(true, hidden, out1, out2, out3);

		} //run

	} //hidden
	out1.close();
	out2.close();
	out3.close();

	return 0;

}
;
