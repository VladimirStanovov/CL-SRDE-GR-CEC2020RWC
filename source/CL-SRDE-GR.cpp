/// Implementation of CL-SRDE-GR algorithm for the
///     Congress on Evolutionary Computation 2020
///     Competition on Real-World Constrained Optimization
/// Author: Vladimir Stanovov (vladimirstanovov@yandex.ru)
///     Reshetnev Siberian State University of Science and Technology
///     Krasnoyarsk, Russian Federation
/// Last change: 23/05/2025

#include <math.h>
#include <iostream>
#include <time.h>
#include <fstream>
#include <random>
#include <mpi.h>
#include <chrono>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <unordered_map>
#include "cec2020rwc.h"

const int ResTsize1 = 57; // number of functions //28 for CEC 2024(2017)
const int ResTsize2 = 2; // number of records per function //2000+1 = 2001 for CEC 2024

using namespace std;
/*typedef std::chrono::high_resolution_clock myclock;
myclock::time_point beginning = myclock::now();
myclock::duration d1 = myclock::now() - beginning;
#ifdef __linux__
    unsigned globalseed = d1.count();
#elif _WIN32
    unsigned globalseed = unsigned(time(NULL));
#else

#endif*/
unsigned globalseed = 2024;
unsigned seed1 = globalseed+0;
unsigned seed2 = globalseed+100;
unsigned seed3 = globalseed+200;
unsigned seed4 = globalseed+300;
unsigned seed5 = globalseed+400;
unsigned seed6 = globalseed+500;
unsigned seed7 = globalseed+600;
std::mt19937 generator_uni_i(seed1);
std::mt19937 generator_uni_r(seed2);
std::mt19937 generator_norm(seed3);
std::mt19937 generator_cachy(seed4);
std::mt19937 generator_uni_i_3(seed5);
std::uniform_int_distribution<int> uni_int(0,32768);
std::uniform_real_distribution<double> uni_real(0.0,1.0);
std::normal_distribution<double> norm_dist(0.0,1.0);
std::cauchy_distribution<double> cachy_dist(0.0,1.0);

int IntRandom(int target) {if(target == 0) return 0; return uni_int(generator_uni_i)%target;}
double Random(double minimal, double maximal){return uni_real(generator_uni_r)*(maximal-minimal)+minimal;}
double NormRand(double mu, double sigma){return norm_dist(generator_norm)*sigma + mu;}
double CachyRand(double mu, double sigma){return cachy_dist(generator_cachy)*sigma+mu;}

void cec17_test_COP(double *x, double *f, double *g,double *h, int nx, int mx,int func_num);

//const int ng_B[28]={1,1,1,2,2,0,0,0,1,0,1,2,3,1,1,1,1,2,2,2,2,3,1,1,1,1,2,2};
//const int nh_B[28]={0,0,1,0,0,6,2,2,1,2,1,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0};
                        //1   2   3  4  5  6  7   8  9  10  11  12  13  14  15  16  17  18 19  20  21  22  23  24  25  26  27 28
//const int border[28] = {100,100,100,10,10,20,50,100,10,100,100,100,100,100,100,100,100,100,50,100,100,100,100,100,100,100,100,50};

const int maxDimension = 158;
const int maxGconstr = 91;
const int maxHconstr = 148;

double *OShift=NULL,*M=NULL,*M1=NULL,*M2=NULL,*y=NULL,*z=NULL,*z1=NULL,*z2=NULL;
int ini_flag=0,n_flag,func_flag,f5_flag;
int stepsFEval[ResTsize2-1];
double ResultsArray[ResTsize1][ResTsize2];
double ResultsArrayG[ResTsize1][ResTsize2][maxGconstr];
double ResultsArrayH[ResTsize1][ResTsize2][maxHconstr];
double ResultsArrayX[ResTsize1][ResTsize2][maxDimension]; //max D = 100
int LastFEcount;
int NFEval = 0;
int MaxFEval = 0;
int GNVars;
double tempF[1];
double tempG[maxGconstr];
double tempH[maxHconstr];
double xopt[100];
double fopt[1];
char buffer[500];
double globalbest;
double globalbestpenalty;
bool globalbestinit;
bool TimeComplexity = false;
double epsilon0001 = 0.0001;
double PRS_mF[16];
double PRS_sF[16];
double PRS_kF[16];
double Cvalglobal = 4;
double EpsIndexglobal = 4;
double GRProbGlobal = 4;
double ConfigGlobal = 4;
double SelPresGlobal = 4;
double CRSGlobal = 0;

struct Params
{
    int TaskN;
    int Type;
    int PopSizeLoop;
    int CvalLoop;
    int EpsIndexLoop;
    int GRProbLoop;
    int ConfigLoop;
    int SelPresLoop;
    int CRSLoop;
    int LPSR_loop;
    int Sel1;
    int Sel2;
    int Sel3;
    int Cval1;
    int Cval2;
    int Cval3;
    int GNVars;
    int MemorySizePar;
    int initF;
    int initCr;
    int pBestStart;
    int pBestEnd;
    int MWLp1;
    int MWLp2;
    int SortCR;
    int TopUpdateType;
    int TopUpdateTypeCR;
    int TopUpdateTypeP;
    int SRType;
    int SRTypeCR;
    int SRTypeP;
};
class Result
{
    public:
    int Node=0;
    int Task=0;
    double ResultTable1[ResTsize1][ResTsize2];
    double ResultTableG1[ResTsize1][ResTsize2][maxGconstr];
    double ResultTableH1[ResTsize1][ResTsize2][maxHconstr];
    double ResultTableX1[ResTsize1][ResTsize2][maxDimension];
    void Copy(Result &Res2, int ResTsize1, int ResTsize2);
    Result(){};
    ~Result(){};
};
void Result::Copy(Result &Res2, int _ResTsize1, int _ResTsize2)
{
    Node = Res2.Node;
    Task = Res2.Task;
    for(int k=0;k!=_ResTsize1;k++)
        for(int j=0;j!=_ResTsize2;j++)
        {
            ResultTable1[k][j] = Res2.ResultTable1[k][j];
            for(int L=0;L!=maxGconstr;L++)
                ResultTableG1[k][j][L] = Res2.ResultTableG1[k][j][L];
            for(int L=0;L!=maxHconstr;L++)
                ResultTableH1[k][j][L] = Res2.ResultTableH1[k][j][L];
			for(int L=0;L!=maxDimension;L++)
                ResultTableX1[k][j][L] = Res2.ResultTableX1[k][j][L];
        }
}
void qSort1(double* Mass, int low, int high)
{
    int i=low;
    int j=high;
    double x=Mass[(low+high)>>1];
    do
    {
        while(Mass[i]<x)    ++i;
        while(Mass[j]>x)    --j;
        if(i<=j)
        {
            double temp=Mass[i];
            Mass[i]=Mass[j];
            Mass[j]=temp;
            i++;    j--;
        }
    } while(i<=j);
    if(low<j)   qSort1(Mass,low,j);
    if(i<high)  qSort1(Mass,i,high);
}
void qSort2int(double* Mass, int* Mass2, int low, int high)
{
    int i=low;
    int j=high;
    double x=Mass[(low+high)>>1];
    do
    {
        while(Mass[i]<x)    ++i;
        while(Mass[j]>x)    --j;
        if(i<=j)
        {
            double temp=Mass[i];
            Mass[i]=Mass[j];
            Mass[j]=temp;
            int temp2=Mass2[i];
            Mass2[i]=Mass2[j];
            Mass2[j]=temp2;
            i++;    j--;
        }
    } while(i<=j);
    if(low<j)   qSort2int(Mass,Mass2,low,j);
    if(i<high)  qSort2int(Mass,Mass2,i,high);
}
const std::string currentDateTime() {
    time_t     now = time(NULL);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);
    return buf;
}
int getNFreeNodes(int* NodeBusy, int world_size)
{
    int counter = 0;
    for(int i=0;i!=world_size;i++)
        counter+=NodeBusy[i];
    return world_size-counter;
}
int getNStartedTasks(vector<int> TaskFinished, int NTasks)
{
    int counter = 0;
    for(int i=0;i!=NTasks;i++)
        if(TaskFinished[i] > 0)
            counter++;
    return counter;
}
int getNFinishedTasks(vector<int> TaskFinished, int NTasks)
{
    int counter = 0;
    for(int i=0;i!=NTasks;i++)
        if(TaskFinished[i] == 2)
            counter++;
    return counter;
}
double getVectorLength(double* Vector, const int Len)
{
    double total = 0;
    for(int i=0;i!=Len;i++)
        total += Vector[i]*Vector[i];
    return sqrt(total);
}
/*void getOptimum(const int func_num)
{
    FILE *fpt=NULL;
	char FileName[30];
    sprintf(FileName, "inputData/shift_data_%d.txt", func_num);
    fpt = fopen(FileName,"r");
    if (fpt==NULL)
        printf("\n Error: Cannot open input file for reading \n");
    for(int k=0;k!=GNVars;k++)
    {
        fscanf(fpt,"%lf",&xopt[k]);
    }
    fclose(fpt);
    cec17_test_COP(xopt, fopt, tempG, tempH, GNVars, 1, func_num);
}*/
double cec_24_constr(double* HostVector, const int func_num)
{
    cec20_func(HostVector,func_num,tempF,tempG,tempH);
    //tempG[0] = 0;tempG[1] = 0;tempG[2] = 0;
    //tempH[0] = 0;tempH[1] = 0;tempH[2] = 0;tempH[3] = 0;tempH[4] = 0;tempH[5] = 0;
    //cec17_test_COP(HostVector, tempF, tempG, tempH, GNVars, 1, func_num);
    NFEval++;
    return tempF[0];
}
double cec_24_totalpenalty(const int func_num, double* PopulG, double* PopulH, double* PopulAllConstr)
{
    for(int i=0;i!=global_gn[func_num-1];i++)
        PopulG[i] = tempG[i];
    for(int i=0;i!=global_hn[func_num-1];i++)
    {
        PopulH[i] = tempH[i];
        if(PopulH[i] < epsilon0001 && PopulH[i] > -epsilon0001)
            PopulH[i] = 0;
    }
    for(int i=0;i!=global_gn[func_num-1]+global_hn[func_num-1];i++)
    {
        if(i < global_gn[func_num-1])
            PopulAllConstr[i] = tempG[i];
        else if(PopulH[i-global_gn[func_num-1]] >= epsilon0001)
            PopulAllConstr[i] = tempH[i-global_gn[func_num-1]];
        else if(PopulH[i-global_gn[func_num-1]] <= -epsilon0001)
            PopulAllConstr[i] = -tempH[i-global_gn[func_num-1]];
        else
            PopulAllConstr[i] = 0;
    }
    double total = 0;
    for(int i=0;i!=global_gn[func_num-1];i++)
        if(tempG[i] > 0)
            total += tempG[i];
    for(int i=0;i!=global_hn[func_num-1];i++)
        if(tempH[i] >= epsilon0001)
            total += tempH[i];
        else if(tempH[i] <= -epsilon0001)
            total += -tempH[i];
    return total/(double(global_gn[func_num-1])+double(global_hn[func_num-1]));
}
void SaveBestValues(int func_num, double* BestG, double* BestH, double* BestInd)
{
    double temp = globalbest;// - fopt[0];
    if(temp <= 1E-8 && globalbestpenalty <= 1E-8 && ResultsArray[func_num-1][ResTsize2-1] == MaxFEval)
    {
        ResultsArray[func_num-1][ResTsize2-1] = NFEval;
    }
    for(int stepFEcount=LastFEcount;stepFEcount<ResTsize2-1;stepFEcount++)
    {
        if(NFEval == stepsFEval[stepFEcount])
        {
            ResultsArray[func_num-1][stepFEcount] = temp;
            for(int i=0;i!=global_gn[func_num-1];i++)
                ResultsArrayG[func_num-1][stepFEcount][i] = BestG[i];
            for(int i=0;i!=global_hn[func_num-1];i++)
                ResultsArrayH[func_num-1][stepFEcount][i] = BestH[i];
			for(int i=0;i!=GNVars;i++)
                ResultsArrayX[func_num-1][stepFEcount][i] = BestInd[i];
            LastFEcount = stepFEcount;
            //cout<<NFEval<<"\t"<<globalbest<<"\t"<<globalbestpenalty<<endl;
        }
    }
}
class Optimizer
{
public:
    int MemorySize;
    int MemoryIter;
    int SuccessFilled;
    int MemoryCurrentIndex;
    int NVars;			    // ðàçìåðíîñòü ïðîñòðàíñòâà
    int NIndsCurrent;
    int NIndsFront;
    int NIndsFrontMax;
    int newNIndsFront;
    int PopulSize;
    int func_num;
    int TheChosenOne;
    int Generation;
    int PFIndex;
    int NConstr;

    double bestfit;
    double SuccessRate;
    double F;       /*ïàðàìåòðû*/
    double Cr;
    double* Right;		    // âåðõíÿÿ ãðàíèöà
    double* Left;		    // íèæíÿÿ ãðàíèöà
    double GRProb;

    double** Popul;	        // ìàññèâ äëÿ ÷àñòèö
    double** PopulG;
    double** PopulH;
    double** PopulAllConstr;
    double** PopulFront;
    double** PopulFrontG;
    double** PopulFrontH;
    double** PopulAllConstrFront;
    double** PopulTemp;
    double** PopulTempG;
    double** PopulTempH;
    double** PopulAllConstrTemp;
    double* FitArr;		// çíà÷åíèÿ ôóíêöèè ïðèãîäíîñòè
    double* FitArrTemp;		// çíà÷åíèÿ ôóíêöèè ïðèãîäíîñòè
    double* FitArrCopy;
    double* FitArrFront;
    double* Trial;
    double* tempSuccessCr;
    double* tempSuccessF;
    double* MemoryCr;
    double* MemoryF;
    double* FitDelta;
    double* Weights;
    double* PenaltyArr;
    double* PenaltyArrFront;
    double* PenaltyTemp;
    double* BestInd;
    double BestG[maxGconstr];
    double BestH[maxHconstr];
    double* BestX;
    double* EpsLevels;
    double* massvector;
    double* tempvector;
    double* epsvector;
    double* xGR;
    double* Cx;
    double** ViolM;
    double* Crvals;
    double* CrvalsTemp;

    int* Indices;
    int* IndicesFront;
    int* IndicesConstr;
    int* IndicesConstrFront;

    void Initialize(int _newNInds, int _newNVars, int _newfunc_num);
    void Clean();
    void MainCycle();
    void UpdateMemoryCr();
    double MeanWL(double* Vector, double* TempWeights);
    void RemoveWorst(int NInds, int NewNInds, double epsilon);
};

void Optimizer::Initialize(int _newNInds, int _newNVars, int _newfunc_num)
{
    NVars = _newNVars;
    NIndsCurrent = _newNInds;
    NIndsFront = _newNInds;
    NIndsFrontMax = _newNInds;
    PopulSize = _newNInds*2;
    Generation = 0;
    TheChosenOne = 0;
    MemorySize = 5;
    MemoryIter = 0;
    SuccessFilled = 0;
    SuccessRate = 0.5;
    func_num = _newfunc_num;
    NConstr = global_gn[func_num-1] + global_hn[func_num-1];
    get_bounds(func_num,Left,Right);
    for(int steps_k=0;steps_k!=ResTsize2-1;steps_k++)
        //stepsFEval[steps_k] = 20000.0/double(ResTsize2-1)*GNVars*(steps_k+1);
        stepsFEval[steps_k] = MaxFEval/double(ResTsize2-1)*(steps_k+1);
    EpsLevels = new double[NConstr];

    Popul = new double*[PopulSize];
    for(int i=0;i!=PopulSize;i++)
        Popul[i] = new double[NVars];
    PopulG = new double*[PopulSize];
    for(int i=0;i!=PopulSize;i++)
        PopulG[i] = new double[maxGconstr];
    PopulH = new double*[PopulSize];
    for(int i=0;i!=PopulSize;i++)
        PopulH[i] = new double[maxHconstr];
    PopulAllConstr = new double*[PopulSize];
    for(int i=0;i!=PopulSize;i++)
        PopulAllConstr[i] = new double[NConstr];

    PopulFront = new double*[NIndsFront];
    for(int i=0;i!=NIndsFront;i++)
        PopulFront[i] = new double[NVars];
    PopulFrontG = new double*[NIndsFront];
    for(int i=0;i!=NIndsFront;i++)
        PopulFrontG[i] = new double[maxGconstr];
    PopulFrontH = new double*[NIndsFront];
    for(int i=0;i!=NIndsFront;i++)
        PopulFrontH[i] = new double[maxHconstr];
    PopulAllConstrFront = new double*[PopulSize];
    for(int i=0;i!=PopulSize;i++)
        PopulAllConstrFront[i] = new double[NConstr];

    PopulTemp = new double*[PopulSize];
    for(int i=0;i!=PopulSize;i++)
        PopulTemp[i] = new double[NVars];
    PopulTempG = new double*[PopulSize];
    for(int i=0;i!=PopulSize;i++)
        PopulTempG[i] = new double[maxGconstr];
    PopulTempH = new double*[PopulSize];
    for(int i=0;i!=PopulSize;i++)
        PopulTempH[i] = new double[maxHconstr];
    PopulAllConstrTemp = new double*[PopulSize];
    for(int i=0;i!=PopulSize;i++)
        PopulAllConstrTemp[i] = new double[NConstr];

    FitArr = new double[PopulSize];
    FitArrTemp = new double[PopulSize];
    FitArrCopy = new double[PopulSize];
    FitArrFront = new double[NIndsFront];

    PenaltyArr = new double[PopulSize];
    PenaltyTemp = new double[PopulSize];
    PenaltyArrFront = new double[PopulSize];

    Weights = new double[PopulSize];
    tempSuccessCr = new double[PopulSize];
    tempSuccessF = new double[PopulSize];
    FitDelta = new double[PopulSize];
    MemoryCr = new double[MemorySize];
    MemoryF = new double[MemorySize];
    Trial = new double[NVars];
    BestInd = new double[NVars];
    massvector=new double[NConstr+1];
    tempvector=new double[NConstr+1];
    epsvector =new double[NConstr+1];

    Indices = new int[PopulSize];
    IndicesFront = new int[PopulSize];
    IndicesConstr = new int[PopulSize];
    IndicesConstrFront = new int[PopulSize];

	for (int i = 0; i<PopulSize; i++)
		for (int j = 0; j<NVars; j++)
			Popul[i][j] = Random(Left[j],Right[j]);
    for(int i=0;i!=PopulSize;i++)
        tempSuccessCr[i] = 0;
    for(int i=0;i!=MemorySize;i++)
        MemoryCr[i] = 1.0;
    for(int i=0;i!=MemorySize;i++)
        MemoryF[i] = 0.3;
    Crvals = new double[PopulSize];
    CrvalsTemp = new double[PopulSize];

    //Gradient Repair
    xGR = new double[NVars];
    Cx = new double[NConstr];
    ViolM = new double*[NVars];
    for(int j=0;j!=NVars;j++)
        ViolM[j] = new double[NConstr];
}
void Optimizer::Clean()
{
    delete Trial;
    delete BestInd;
    for(int i=0;i!=PopulSize;i++)
    {
        delete Popul[i];
        delete PopulG[i];
        delete PopulH[i];
        delete PopulTemp[i];
        delete PopulTempG[i];
        delete PopulTempH[i];
        delete PopulAllConstr[i];
        delete PopulAllConstrTemp[i];
    }
    for(int i=0;i!=NIndsFrontMax;i++)
    {
        delete PopulFront[i];
        delete PopulFrontG[i];
        delete PopulFrontH[i];
    }
    delete Popul;
    delete PopulG;
    delete PopulH;
    delete PopulTemp;
    delete PopulTempG;
    delete PopulTempH;
    delete PopulAllConstr;
    delete PopulAllConstrTemp;
    delete PopulAllConstrFront;
    delete PopulFront;
    delete PopulFrontG;
    delete PopulFrontH;
    delete FitArr;
    delete FitArrTemp;
    delete FitArrCopy;
    delete FitArrFront;
    delete PenaltyArr;
    delete PenaltyArrFront;
    delete PenaltyTemp;
    delete Indices;
    delete IndicesFront;
    delete IndicesConstr;
    delete IndicesConstrFront;
    delete tempSuccessCr;
    delete tempSuccessF;
    delete FitDelta;
    delete MemoryCr;
    delete MemoryF;
    delete Weights;
    delete EpsLevels;
    delete massvector;
    delete tempvector;
    delete epsvector;
    delete xGR;
    for(int j=0;j!=NVars;j++)
        delete ViolM[j];
    delete ViolM;
    delete Cx;
    delete Crvals;
    delete CrvalsTemp;
}
void Optimizer::UpdateMemoryCr()
{
    if(SuccessFilled != 0)
    {
        MemoryCr[MemoryIter] = 0.5*(MeanWL(tempSuccessCr,FitDelta) + MemoryCr[MemoryIter]);
        MemoryF[MemoryIter] = 0.5*(MeanWL(tempSuccessF,FitDelta) + MemoryF[MemoryIter]);
        MemoryIter = (MemoryIter+1)%MemorySize;
    }
}
double Optimizer::MeanWL(double* Vector, double* TempWeights)
{
    double SumWeight = 0;
    double SumSquare = 0;
    double Sum = 0;
    for(int i=0;i!=SuccessFilled;i++)
        SumWeight += TempWeights[i];
    for(int i=0;i!=SuccessFilled;i++)
        Weights[i] = TempWeights[i]/SumWeight;
    for(int i=0;i!=SuccessFilled;i++)
        SumSquare += Weights[i]*Vector[i]*Vector[i];
    for(int i=0;i!=SuccessFilled;i++)
        Sum += Weights[i]*Vector[i];
    if(fabs(Sum) > 1e-8)
        return SumSquare/Sum;
    else
        return 1.0;
}

void Optimizer::RemoveWorst(int _NIndsFront, int _newNIndsFront, double epsilon)
{
    int PointsToRemove = _NIndsFront - _newNIndsFront;
    for(int L=0;L!=PointsToRemove;L++)
    {
        double maxfitFront = FitArrFront[0];
        for(int i=0;i!=NIndsFront-L;i++)
            maxfitFront = max(maxfitFront,FitArrFront[i]);
        double maxfit = FitArrFront[0];
        int WorstNum = 0;
        for(int i=0;i!=NIndsFront-L;i++)
        {
            if(PenaltyArrFront[i] > epsilon)
            {
                if(maxfitFront + 1.0 + PenaltyArrFront[i] > maxfit)
                {
                    maxfit = maxfitFront + 1.0 + PenaltyArrFront[i];
                    WorstNum = i;
                }
            }
            else
            {
                if(FitArrFront[i] > maxfit)
                {
                    maxfit = FitArrFront[i];
                    WorstNum = i;
                }
            }
        }
        for(int i=WorstNum;i!=_NIndsFront-1;i++)
        {
            for(int j=0;j!=NVars;j++)
                PopulFront[i][j] = PopulFront[i+1][j];
            for(int j=0;j!=global_gn[func_num-1];j++)
                PopulFrontG[i][j] = PopulFrontG[i+1][j];
            for(int j=0;j!=global_hn[func_num-1];j++)
                PopulFrontH[i][j] = PopulFrontH[i+1][j];
            for(int j=0;j!=NConstr;j++)
                PopulAllConstrFront[i][j] = PopulAllConstrFront[i+1][j];
            FitArrFront[i] = FitArrFront[i+1];
            PenaltyArrFront[i] = PenaltyArrFront[i+1];
        }
    }
}
void Optimizer::MainCycle()
{
    MatrixXd NabC(NConstr,NVars);
    MatrixXd invNabC(NVars,NConstr);
    MatrixXd delC(NConstr,1);
    MatrixXd deltaX(NVars,1);
    int NumG = global_gn[func_num-1];
    int NumH = global_hn[func_num-1];
    double epsilon = 0.0001;
    double epsilonf = 0.0001;
    double ECutoffParam = 0.8;
    double MaxViol = 0;
    double SuccessRateRE = 0.5;
    double tau = 0.1;
    double alpha = 0.5;
    double cp = 2;
    double LastEpsilon = 0;
    double phiMax = 0;
    vector<double> FitTemp2;
    for(int IndIter=0;IndIter<NIndsFront;IndIter++)
    {
        FitArr[IndIter] = cec_24_constr(Popul[IndIter],func_num);
        PenaltyArr[IndIter] = cec_24_totalpenalty(func_num,PopulG[IndIter],PopulH[IndIter],PopulAllConstr[IndIter]);
        if(globalbestinit || PenaltyArr[IndIter] > MaxViol)
        {
            MaxViol = PenaltyArr[IndIter];
            LastEpsilon = MaxViol;
            phiMax = MaxViol;
        }
        if(!globalbestinit ||
           ((FitArr[IndIter] <= globalbest) && PenaltyArr[IndIter] <= 0) ||
           ((PenaltyArr[IndIter] <= globalbestpenalty) && PenaltyArr[IndIter] > 0))
        {
            globalbest = FitArr[IndIter];
            globalbestpenalty = PenaltyArr[IndIter];
            globalbestinit = true;
            bestfit = FitArr[IndIter];
            for(int j=0;j!=NVars;j++)
                BestInd[j] = Popul[IndIter][j];
            for(int j=0;j!=NumG;j++)
                BestG[j] = PopulG[IndIter][j];
            for(int j=0;j!=NumH;j++)
                BestH[j] = PopulH[IndIter][j];
        }
        SaveBestValues(func_num,BestG,BestH,BestInd);
    }
    for(int i=0;i!=NIndsFront;i++)
    {
        for(int j=0;j!=NVars;j++)
            PopulFront[i][j] = Popul[i][j];
        for(int j=0;j!=NumG;j++)
            PopulFrontG[i][j] = PopulG[i][j];
        for(int j=0;j!=NumH;j++)
            PopulFrontH[i][j] = PopulH[i][j];
        for(int j=0;j!=NConstr;j++)
            PopulAllConstrFront[i][j] = PopulAllConstr[i][j];
        FitArrFront[i] = FitArr[i];
        PenaltyArrFront[i] = PenaltyArr[i];
    }
    PFIndex = 0;
    double cp_e0 = 2;//(-log(MaxViol)-6.0)/(log(1.0-ECutoffParam));
    double EIndexParam = 0.01*EpsIndexglobal;
    while(NFEval < MaxFEval)
    {
        double meanF;
        if(ConfigGlobal == 0)
            meanF = 0.4+tanh(SuccessRate*5)*0.25;//max(0.0,pow(SuccessRate,1.0/3.0));//
        else if(ConfigGlobal == 1)
            meanF = max(0.0,pow(SuccessRate,1.0/3.0));//
        else if(ConfigGlobal == 2)
            meanF = max(0.0,pow(SuccessRate,1.0/3.0));//max(0.0,pow(SuccessRate,1.0/3.0));//
        else if(ConfigGlobal == 3)
            meanF = 0; // use memoryF !
        double sigmaF = 0.02;//0.05;//
        if(ConfigGlobal >= 2)
            sigmaF = 0.1;

        int epsilonindex = (NIndsFront*EIndexParam*
                        (1.0-(double)NFEval/(double)MaxFEval)*
                        (1.0-(double)NFEval/(double)MaxFEval));
        int epsilonfindex = (EIndexParam*NIndsFront*
                        (1.0-(double)NFEval/(double)MaxFEval)*
                        (1.0-(double)NFEval/(double)MaxFEval));

        int psizeval;
        if(ConfigGlobal == 0)
            psizeval = max(2,int(NIndsFront*0.7*exp(-SuccessRate*7)));//max(2,int(0.3*NIndsFront));
        else if(ConfigGlobal == 1)
            psizeval = max(2,int(0.3*NIndsFront));
        else if(ConfigGlobal == 2)
            psizeval = max(2,int(NIndsFront*0.7*exp(-SuccessRateRE*7)));//max(2,int(0.3*NIndsFront));
        else if(ConfigGlobal == 3)
            psizeval = max(2,int(0.3*NIndsFront)); // use memoryF !

        double minfit = FitArrFront[0];
        double maxfit = FitArrFront[0];
        for(int i=0;i!=NIndsFront;i++)
        {
            FitArrCopy[i] = FitArrFront[i];
            Indices[i] = i;
            if(FitArrFront[i] >= maxfit)
                maxfit = FitArrFront[i];
            if(FitArrFront[i] <= minfit)
                minfit = FitArrFront[i];
        }
        if(minfit != maxfit)
            qSort2int(FitArrCopy,Indices,0,NIndsFront-1);
        epsilonf = FitArrCopy[epsilonfindex];
        if(NFEval > ECutoffParam*MaxFEval)
            epsilonf = minfit;

        for(int C=0;C!=NConstr;C++)
        {
            for(int i=0;i!=NIndsFront;i++)
            {
                FitArrCopy[i] = PopulAllConstrFront[i][C];
                if(FitArrCopy[i] < epsilon0001)
                    FitArrCopy[i] = 0;
                IndicesConstr[i] = i;
            }
            double penaltymax = FitArrCopy[0];
            double penaltymin = FitArrCopy[0];
            for(int i=0;i!=NIndsFront;i++)
            {
                if(FitArrCopy[i] >= penaltymax)
                    penaltymax = FitArrCopy[i];
                if(FitArrCopy[i] <= penaltymin)
                    penaltymin = FitArrCopy[i];
            }
            if(penaltymin != penaltymax)
                qSort2int(FitArrCopy,IndicesConstr,0,NIndsFront-1);
            if(EpsLevels[C] > FitArrCopy[epsilonindex] || Generation == 0)
                EpsLevels[C] = FitArrCopy[epsilonindex];
            if(NFEval > ECutoffParam*MaxFEval)
                EpsLevels[C] = 0.0;
        }

        int prand = 0;
        int Rand1 = 0;
        int Rand2 = 0;

        double FeasRate = 0;
        for(int i=0;i!=NIndsFront;i++)
        {
            if(PenaltyArrFront[i] < epsilon0001)
                FeasRate += 1;
        }
        FeasRate = FeasRate / double(NIndsFront);

        minfit = PenaltyArrFront[0];
        maxfit = PenaltyArrFront[0];
        for(int i=0;i!=NIndsFront;i++)
        {
            FitArrCopy[i] = PenaltyArrFront[i];
            IndicesConstr[i] = i;
            maxfit = max(maxfit,PenaltyArrFront[i]);
            minfit = min(minfit,PenaltyArrFront[i]);
        }
        if(minfit != maxfit)
            qSort2int(FitArrCopy,IndicesConstr,0,NIndsFront-1);
        phiMax = maxfit;
        //epsilon = FitArrCopy[epsilonindex];
        if(Cvalglobal == 0)
            epsilon = FitArrCopy[epsilonindex];
        else if(Cvalglobal == 1)
            epsilon = MaxViol*pow((1.0-(double)NFEval/(double)MaxFEval),cp_e0);
        else if(Cvalglobal == 2)
        {
            if(FeasRate < alpha)
                epsilon = LastEpsilon*pow((1.0-(double)NFEval/(double)MaxFEval),cp);
            else
                epsilon = (1.0+tau)*phiMax;
        }
        if((double)NFEval/(double)MaxFEval > ECutoffParam)
            epsilon = 0.0;

        double maxfitPop = FitArr[0];
        for(int i=0;i!=NIndsFront;i++)
            maxfitPop = max(maxfitPop,FitArr[i]);
        //get indices for front
        for(int i=0;i!=NIndsFront;i++)
        {
            if(PenaltyArr[i] > epsilon)
                FitArrCopy[i] = maxfitPop + 1.0 + PenaltyArr[i];
            else
                FitArrCopy[i] = FitArr[i];
            if(i == 0)
            {
                minfit = FitArrCopy[0];
                maxfit = FitArrCopy[0];
            }
            Indices[i] = i;
            maxfit = max(maxfit,FitArr[i]);
            minfit = min(minfit,FitArr[i]);
        }
        if(minfit != maxfit)
            qSort2int(FitArrCopy,Indices,0,NIndsFront-1);

        //ranks for selective pressure
        FitTemp2.resize(NIndsFront);
        for(int i=0;i!=NIndsFront;i++)
            FitTemp2[i] = exp(-double(i)/double(NIndsFront)*SelPresGlobal);
        std::discrete_distribution<int> ComponentSelectorFront (FitTemp2.begin(),FitTemp2.end());


        double maxfitFront = FitArrFront[0];
        for(int i=0;i!=NIndsFront;i++)
            maxfitFront = max(maxfitFront,FitArrFront[i]);
        //get indices for popul
        for(int i=0;i!=NIndsFront;i++)
        {
            if(PenaltyArrFront[i] > epsilon)
                FitArrCopy[i] = maxfitFront + 1.0 + PenaltyArrFront[i];
            else
                FitArrCopy[i] = FitArrFront[i];
            if(i == 0)
            {
                minfit = FitArrCopy[0];
                maxfit = FitArrCopy[0];
            }
            IndicesFront[i] = i;
            maxfit = max(maxfit,FitArrFront[i]);
            minfit = min(minfit,FitArrFront[i]);
        }
        if(minfit != maxfit)
            qSort2int(FitArrCopy,IndicesFront,0,NIndsFront-1);

        for(int IndIter=0;IndIter<NIndsFront;IndIter++)
        {
            Cr = NormRand(MemoryCr[MemoryCurrentIndex],0.1);
            Cr = min(max(Cr,0.0),1.0);
            CrvalsTemp[IndIter] = Cr;
        }
        qSort1(CrvalsTemp,0,NIndsFront-1);
        for(int IndIter=0;IndIter<NIndsFront;IndIter++)
        {
            Crvals[IndicesFront[IndIter]] = CrvalsTemp[IndIter];
        }
        for(int IndIter=0;IndIter<NIndsFront;IndIter++)
        {
            double maxfitFront = FitArrFront[0];
            for(int i=0;i!=NIndsFront;i++)
                maxfitFront = max(maxfitFront,FitArrFront[i]);
            //get indices for popul
            for(int i=0;i!=NIndsFront;i++)
            {
                if(PenaltyArrFront[i] > epsilon)
                    FitArrCopy[i] = maxfitFront + 1.0 + PenaltyArrFront[i];
                else
                    FitArrCopy[i] = FitArrFront[i];
                if(i == 0)
                {
                    minfit = FitArrCopy[0];
                    maxfit = FitArrCopy[0];
                }
                IndicesFront[i] = i;
                maxfit = max(maxfit,FitArrFront[i]);
                minfit = min(minfit,FitArrFront[i]);
            }
            if(minfit != maxfit)
                qSort2int(FitArrCopy,IndicesFront,0,NIndsFront-1);

            TheChosenOne = IntRandom(NIndsFront);
            MemoryCurrentIndex = IntRandom(MemorySize);
            do
                prand = Indices[IntRandom(psizeval)];
            while(prand == TheChosenOne);
            do
                Rand1 = IndicesFront[ComponentSelectorFront(generator_uni_i_3)];
            while(Rand1 == prand);
            do
                Rand2 = Indices[IntRandom(NIndsFront)];
            while(Rand2 == prand || Rand2 == Rand1);


            double meanF2 = meanF;
            if(ConfigGlobal == 3)//if(Random(0,1) < 0.1*Cvalglobal)
                meanF2 = MemoryF[MemoryCurrentIndex];
            do
                F = NormRand(meanF2,sigmaF);
            while(F < 0.0 || F > 1.0);
            double F2 = F;

            if(CRSGlobal == 0)
            {
                Cr = NormRand(MemoryCr[MemoryCurrentIndex],0.1);
                Cr = min(max(Cr,0.0),1.0);
            }
            else
            {
                Cr = Crvals[TheChosenOne];
            }
            int WillCrossover = IntRandom(NVars);
            double ActualCr = 0;
            for(int j=0;j!=NVars;j++)
            {
                if(Random(0,1) < Cr || WillCrossover == j)
                {
                    Trial[j] = PopulFront[TheChosenOne][j] + F*(Popul[prand][j] - PopulFront[TheChosenOne][j]) + F2*(PopulFront[Rand1][j] - Popul[Rand2][j]);
                    if(Trial[j] < Left[j])
                        Trial[j] = (PopulFront[TheChosenOne][j] + Left[j])*0.5;
                    if(Trial[j] > Right[j])
                        Trial[j] = (PopulFront[TheChosenOne][j] + Right[j])*0.5;
                    ActualCr ++;
                }
                else
                    Trial[j] = PopulFront[TheChosenOne][j];
                PopulTemp[IndIter][j] = Trial[j];
            }
            ActualCr = ActualCr / double(NVars);

            double eta = 1e-4;
            GRProb = 0.01*GRProbGlobal;
            if(Random(0,1) < GRProb && PenaltyArrFront[TheChosenOne] > 0)
            {
                //cout.precision(18);
                for(int j=0;j!=NVars;j++)
                {
                    xGR[j] = PopulFront[TheChosenOne][j];
                    //cout<<xGR[j]<<",\t";
                }
                //cout<<endl;
                for(int L=0;L!=global_gn[func_num-1];L++)
                {
                    Cx[L] = PopulFrontG[TheChosenOne][L];
                    delC(L) = max(0.0,PopulFrontG[TheChosenOne][L]);
                    //cout<<Cx[L]<<",\t";
                }
                for(int L=global_gn[func_num-1];L!=global_gn[func_num-1]+global_hn[func_num-1];L++)
                {
                    Cx[L] = PopulFrontH[TheChosenOne][L-global_gn[func_num-1]];
                    delC(L) = PopulFrontH[TheChosenOne][L-global_gn[func_num-1]];
                    //cout<<Cx[L]<<",\t";
                }
                //cout<<endl;
                //cout<<delC<<endl;
                for(int k=0;k!=NVars;k++)
                {
                    xGR[k] += eta;
                    FitArrTemp[IndIter] = cec_24_constr(xGR,func_num);
                    PenaltyTemp[IndIter] = cec_24_totalpenalty(func_num,PopulTempG[IndIter],PopulTempH[IndIter],PopulAllConstrTemp[IndIter]);
                    if(!globalbestinit ||
                       ((FitArrTemp[IndIter] <= globalbest) && PenaltyTemp[IndIter] <= 0) ||
                        ((PenaltyTemp[IndIter] <= globalbestpenalty) && PenaltyTemp[IndIter] > 0))
                    {
                        globalbest = FitArrTemp[IndIter];
                        globalbestpenalty = PenaltyTemp[IndIter];
                        globalbestinit = true;
                        bestfit = FitArrTemp[IndIter];
                        for(int j=0;j!=NVars;j++)
                            BestInd[j] = PopulTemp[IndIter][j];
                        for(int j=0;j!=NumG;j++)
                            BestG[j] = PopulTempG[IndIter][j];
                        for(int j=0;j!=NumH;j++)
                            BestH[j] = PopulTempH[IndIter][j];
                        //cout<<NFEval<<"\t"<<globalbest<<"\t"<<globalbestpenalty<<endl;
                    }
                    SaveBestValues(func_num,BestG,BestH,BestInd);
                    NFEval--;
                    for(int L=0;L!=global_gn[func_num-1];L++)
                    {
                        ViolM[k][L] = PopulTempG[IndIter][L];
                    }
                    for(int L=global_gn[func_num-1];L!=global_gn[func_num-1]+global_hn[func_num-1];L++)
                    {
                        ViolM[k][L] = PopulTempH[IndIter][L-global_gn[func_num-1]];
                    }
                    //for(int L=0;L!=NConstr;L++)
                        //ViolM[k][L] = PopulAllConstrTemp[IndIter][L];
                    xGR[k] = PopulFront[TheChosenOne][k];
                }
                for(int k=0;k!=NVars;k++)
                {
                    for(int L=0;L!=NConstr;L++)
                        NabC(L,k) = 1.0/eta*(ViolM[k][L] - Cx[L]);

                }
                //cout<<NabC<<endl;
                // Moore-Penrose pseudo-inverse from: https://gist.github.com/pshriwise/67c2ae78e5db3831da38390a8b2a209f
                Eigen::JacobiSVD<MatrixXd> svd(NabC ,Eigen::ComputeThinU | Eigen::ComputeThinV);
                double tolerance = std::numeric_limits<double>::epsilon() * std::max(NabC.cols(), NabC.rows()) *svd.singularValues().array().abs()(0);
                invNabC = svd.matrixV() *  (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal() * svd.matrixU().adjoint();

                deltaX = -invNabC*delC;

                //cout<<deltaX<<endl;
                for(int j=0;j!=NVars;j++)
                {
                    Trial[j] = xGR[j] + deltaX(j);
                    if(Trial[j] < Left[j])
                        Trial[j] = (PopulFront[TheChosenOne][j] + Left[j])*0.5;
                    if(Trial[j] > Right[j])
                        Trial[j] = (PopulFront[TheChosenOne][j] + Right[j])*0.5;
                    PopulTemp[IndIter][j] = Trial[j];
                }
                NFEval++;
            }

            FitArrTemp[IndIter] = cec_24_constr(PopulTemp[IndIter],func_num);
            PenaltyTemp[IndIter] = cec_24_totalpenalty(func_num,PopulTempG[IndIter],PopulTempH[IndIter],PopulAllConstrTemp[IndIter]);
            if(!globalbestinit ||
               ((FitArrTemp[IndIter] <= globalbest) && PenaltyTemp[IndIter] <= 0) ||
                ((PenaltyTemp[IndIter] <= globalbestpenalty) && PenaltyTemp[IndIter] > 0))
            {
                globalbest = FitArrTemp[IndIter];
                globalbestpenalty = PenaltyTemp[IndIter];
                globalbestinit = true;
                bestfit = FitArrTemp[IndIter];
                for(int j=0;j!=NVars;j++)
                    BestInd[j] = PopulTemp[IndIter][j];
                for(int j=0;j!=NumG;j++)
                    BestG[j] = PopulTempG[IndIter][j];
                for(int j=0;j!=NumH;j++)
                    BestH[j] = PopulTempH[IndIter][j];
                //cout<<NFEval<<"\t"<<globalbest<<"\t"<<globalbestpenalty<<endl;
            }
            //if(globalbest == 0)
              //  cout<<"WARINING!"<<endl;

            double improvement = 0;
            bool change = false;
            if(Random(0,1) < 1.0)
            {
                double temppenalty = PenaltyTemp[IndIter];
                double frontpenalty = PenaltyArrFront[TheChosenOne];
                bool goodtemp = false;
                if(temppenalty <= epsilon) //new is epsilon-feasible
                {
                    temppenalty = 0;
                    goodtemp = true;
                }
                if(frontpenalty <= epsilon) // current is epsilon-feasible
                    frontpenalty = 0;
                if(
                   ((goodtemp)&&(FitArrTemp[IndIter]<=FitArrFront[TheChosenOne])) || //new is epsilon-feasible, and with better fitness
                   ((temppenalty==frontpenalty) && (FitArrTemp[IndIter]<=FitArrFront[TheChosenOne])) //new has same penalty and better fitness
                   )
                {
                    change = true;
                    improvement = FitArrFront[TheChosenOne] - FitArrTemp[IndIter];
                }
                else if(temppenalty < frontpenalty) //both are epsilon-infeasible, but new has smaller violation
                {
                    change = true;
                    improvement = frontpenalty - temppenalty;
                }
            }
            else
            {
                ////////////////////////////////////////////////epsilonf
                double temppenalty = PenaltyTemp[IndIter];
                double frontpenalty = PenaltyArrFront[TheChosenOne];
                bool goodtemp = false;
                if(FitArrTemp[IndIter] <= epsilonf)
                {
                    temppenalty = 0;
                    goodtemp = true;
                }
                if(FitArrFront[TheChosenOne] <= epsilonf)
                    frontpenalty = 0;
                if((goodtemp && FitArrTemp[IndIter] <= FitArrFront[TheChosenOne] ) ||
                   ((temppenalty==frontpenalty) && (FitArrTemp[IndIter]<=FitArrFront[TheChosenOne])))
                {
                    change = true;
                    improvement = FitArrFront[TheChosenOne] - FitArrTemp[IndIter];
                }
                else if(temppenalty < frontpenalty)
                {
                    change = true;
                    improvement = frontpenalty - temppenalty;
                }
                ////////////////////////////////////////////////epsilonf
            }
            if(change)
            {
                for(int j=0;j!=NVars;j++)
                {
                    Popul[NIndsCurrent+SuccessFilled][j] = PopulTemp[IndIter][j];
                    PopulFront[PFIndex][j] = PopulTemp[IndIter][j];
                }
                for(int j=0;j!=NumG;j++)
                {
                    PopulG[NIndsCurrent+SuccessFilled][j] = PopulTempG[IndIter][j];
                    PopulFrontG[PFIndex][j] = PopulTempG[IndIter][j];
                }
                for(int j=0;j!=NumH;j++)
                {
                    PopulH[NIndsCurrent+SuccessFilled][j] = PopulTempH[IndIter][j];
                    PopulFrontH[PFIndex][j] = PopulTempH[IndIter][j];
                }
                for(int j=0;j!=NConstr;j++)
                {
                    PopulAllConstr[NIndsCurrent+SuccessFilled][j] = PopulAllConstrTemp[IndIter][j];
                    PopulAllConstrFront[PFIndex][j] = PopulAllConstrTemp[IndIter][j];
                }
                FitArr[NIndsCurrent+SuccessFilled] = FitArrTemp[IndIter];
                FitArrFront[PFIndex] = FitArrTemp[IndIter];
                PenaltyArr[NIndsCurrent+SuccessFilled] = PenaltyTemp[IndIter];
                PenaltyArrFront[PFIndex] = PenaltyTemp[IndIter];
                tempSuccessCr[SuccessFilled] = ActualCr;//Cr
                tempSuccessF[SuccessFilled] = F;
                FitDelta[SuccessFilled] = improvement;
                SuccessFilled++;
                PFIndex = (PFIndex + 1)%NIndsFront;
            }
            SaveBestValues(func_num,BestG,BestH,BestInd);
        }
        SuccessRate = double(SuccessFilled)/double(NIndsFront);
        SuccessRateRE = 0.7*SuccessRateRE + SuccessRate*0.3;
        newNIndsFront = int(double(4-NIndsFrontMax)/double(MaxFEval)*NFEval + NIndsFrontMax);
        RemoveWorst(NIndsFront,newNIndsFront,epsilon);
        NIndsFront = newNIndsFront;
        UpdateMemoryCr();
        NIndsCurrent = NIndsFront + SuccessFilled;
        SuccessFilled = 0;
        Generation++;
        if(NIndsCurrent > NIndsFront)
        {
            double maxfitPop = FitArr[0];
            for(int i=0;i!=NIndsCurrent;i++)
                maxfitPop = max(maxfitPop,FitArr[i]);

            for(int i=0;i!=NIndsCurrent;i++)
            {
                if(PenaltyArr[i] > epsilon)
                    FitArrCopy[i] = maxfitPop + 1.0 + PenaltyArr[i];
                else
                    FitArrCopy[i] = FitArr[i];
                if(i == 0)
                {
                    minfit = FitArrCopy[0];
                    maxfit = FitArrCopy[0];
                }
                Indices[i] = i;
                maxfit = max(maxfit,FitArr[i]);
                minfit = min(minfit,FitArr[i]);
            }
            if(minfit != maxfit)
                qSort2int(FitArrCopy,Indices,0,NIndsCurrent-1);

            NIndsCurrent = NIndsFront;
            for(int i=0;i!=NIndsCurrent;i++)
            {
                for(int j=0;j!=NVars;j++)
                    PopulTemp[i][j] = Popul[Indices[i]][j];
                for(int j=0;j!=NumG;j++)
                    PopulTempG[i][j] = PopulG[Indices[i]][j];
                for(int j=0;j!=NumH;j++)
                    PopulTempH[i][j] = PopulH[Indices[i]][j];
                for(int j=0;j!=NConstr;j++)
                    PopulAllConstrTemp[i][j] = PopulAllConstr[Indices[i]][j];
                FitArrTemp[i] = FitArr[Indices[i]];
                PenaltyTemp[i] = PenaltyArr[Indices[i]];
            }
            for(int i=0;i!=NIndsCurrent;i++)
            {
                for(int j=0;j!=NVars;j++)
                    Popul[i][j] = PopulTemp[i][j];
                for(int j=0;j!=NumG;j++)
                    PopulG[i][j] = PopulTempG[i][j];
                for(int j=0;j!=NumH;j++)
                    PopulH[i][j] = PopulTempH[i][j];
                for(int j=0;j!=NConstr;j++)
                    PopulAllConstr[i][j] = PopulAllConstrTemp[i][j];
                FitArr[i] = FitArrTemp[i];
                PenaltyArr[i] = PenaltyTemp[i];
            }
        }
    }
}

int main(int argc, char** argv)
{
    string addname = "";
    if(argc > 1)
        addname = argv[1];
    unsigned t0g=clock(),t1g;

    int world_size,world_rank,name_len,TotalNRuns;
    TotalNRuns = 25;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(processor_name, &name_len);

    seed1 += world_rank;
    seed2 += world_rank;
    seed3 += world_rank;
    seed4 += world_rank;
    seed5 += world_rank;

    generator_uni_i.seed(seed1);
    generator_uni_r.seed(seed2);
    generator_norm.seed(seed3);
    generator_cachy.seed(seed4);
    generator_uni_i_3.seed(seed5);
    cout<<world_rank<<"\t"<<seed1<<endl;

    int NTasks = 0;
    vector<Params> PRS;
    vector<int> TaskFinished;
    vector<Result> AllResults;
    vector<string> FilePath;
    vector<string> FilePathC;
    vector<string> FilePathC2;
    vector<string> FilePathShort;
    vector<string> FilePathShortC;
    int ResultSize = sizeof(Result);
    int* NodeBusy = new int[world_size-1];
    int* NodeTask = new int[world_size-1];
    for(int i=0;i!=world_size-1;i++)
    {
        NodeBusy[i] = 0;
        NodeTask[i] = -1;
    }
    Params tempPRS;
    Result tempResult;
    string tempFilePath;
    string tempFilePathC;
    string tempFilePathC2;
    string tempFilePathShort;
    string tempFilePathShortC;
    int TaskCounter = 0;
    int DiffPRSCounter = 0;
    ofstream foutTasks;
    int MinGNVarsIter = 0;
    int MaxGNVarsIter = 1;

    for(int GNVarsIter = MinGNVarsIter;GNVarsIter!=MaxGNVarsIter;GNVarsIter++)    {
    for(int PopSizeLoop=80;PopSizeLoop<81;PopSizeLoop+=10)    {
    for(int CvalLoop=0;CvalLoop<1;CvalLoop+=1)    {
    for(int EpsIndexLoop=25;EpsIndexLoop<26;EpsIndexLoop+=5)    {
    for(int GRProbLoop=0;GRProbLoop<1;GRProbLoop+=2)    {
    for(int ConfigLoop=0;ConfigLoop<1;ConfigLoop+=1)    {
    for(int SelPresLoop=9;SelPresLoop<10;SelPresLoop+=1)    {
    for(int CRSLoop=1;CRSLoop<2;CRSLoop+=1)    {
            //if(SelPresLoop%2 == 0 && SelPresLoop < 9)
              //  continue;

    GNVars = 10;

    DiffPRSCounter++;
    for (int RunN = 0;RunN!=TotalNRuns;RunN++)    {
        tempPRS.TaskN = TaskCounter;
        tempPRS.PopSizeLoop = PopSizeLoop;
        tempPRS.CvalLoop = CvalLoop;
        tempPRS.EpsIndexLoop = EpsIndexLoop;
        tempPRS.GRProbLoop = GRProbLoop;
        tempPRS.ConfigLoop = ConfigLoop;
        tempPRS.SelPresLoop = SelPresLoop;
        tempPRS.CRSLoop = CRSLoop;
        tempPRS.LPSR_loop = 6;
        tempPRS.GNVars = GNVars;

        PRS.push_back(tempPRS);
        //cout<<"Run\t"<<RunN<<"\tst"<<endl;
        AllResults.push_back(tempResult);
        //cout<<"Run\t"<<RunN<<"\tend"<<endl;
        TaskFinished.push_back(0);

        string folder;
        folder = "../RE_25_RWC2020_Raw_results";
        //folder = "/home/mpiscil/cloud/DE_24_Results/RE_25_RWC2020";
        //folder = "/media/sf_VVS/Documents/CEC_2020_to_c/RE_25_RWC2020";

        struct stat st = {0};
        if(world_rank == 0)
        {
            if (stat(folder.c_str(), &st) == -1)
                mkdir(folder.c_str(), 0777);
        }
        sprintf(buffer, "/Results_D%d_L%d_PS%d_CV%d_EI%d_GR%d_CF%d_SP%d_CRS%d_%s.txt",
            GNVars,tempPRS.LPSR_loop,PopSizeLoop,CvalLoop,EpsIndexLoop,GRProbLoop,ConfigLoop,SelPresLoop,CRSLoop,addname.c_str());
        tempFilePath = folder + buffer;
        FilePath.push_back(tempFilePath);
        if(world_rank == 0)
            cout<<world_rank<<" "<<TaskCounter<<"\t"<<buffer<<endl;
        sprintf(buffer, "/Results_C_D%d_L%d_PS%d_CV%d_EI%d_GR%d_CF%d_SP%d_CRS%d_%s.txt",
            GNVars,tempPRS.LPSR_loop,PopSizeLoop,CvalLoop,EpsIndexLoop,GRProbLoop,ConfigLoop,SelPresLoop,CRSLoop,addname.c_str());
        tempFilePathC = folder + buffer;
        FilePathC.push_back(tempFilePathC);
        //if(world_rank == 0)
            //cout<<world_rank<<" "<<TaskCounter<<"\t"<<buffer<<endl;
        sprintf(buffer, "/Results_C2_D%d_L%d_PS%d_CV%d_EI%d_GR%d_CF%d_SP%d_CRS%d_%s.txt",
            GNVars,tempPRS.LPSR_loop,PopSizeLoop,CvalLoop,EpsIndexLoop,GRProbLoop,ConfigLoop,SelPresLoop,CRSLoop,addname.c_str());
        tempFilePathC2 = folder + buffer;
        FilePathC2.push_back(tempFilePathC2);
		sprintf(buffer, "/Results_Short_D%d_L%d_PS%d_CV%d_EI%d_GR%d_CF%d_SP%d_CRS%d_%s.txt",
            GNVars,tempPRS.LPSR_loop,PopSizeLoop,CvalLoop,EpsIndexLoop,GRProbLoop,ConfigLoop,SelPresLoop,CRSLoop,addname.c_str());
        tempFilePathShort = folder + buffer;
        FilePathShort.push_back(tempFilePathShort);
		sprintf(buffer, "/Results_ShortC_D%d_L%d_PS%d_CV%d_EI%d_GR%d_CF%d_SP%d_CRS%d_%s.txt",
            GNVars,tempPRS.LPSR_loop,PopSizeLoop,CvalLoop,EpsIndexLoop,GRProbLoop,ConfigLoop,SelPresLoop,CRSLoop,addname.c_str());
        tempFilePathShortC = folder + buffer;
        FilePathShortC.push_back(tempFilePathShortC);
        //if(world_rank == 0)
            //cout<<world_rank<<" "<<TaskCounter<<"\t"<<buffer<<endl;
        TaskCounter++;
    }}}}}}}}}

    NTasks = TaskCounter;
    cout<<"Total\t"<<NTasks<<"\ttasks"<<endl;
    int PRSsize = sizeof(tempPRS);
    vector<int> ResSavedPerPRS;
    ResSavedPerPRS.resize(DiffPRSCounter);
    cout<<"ResSavedPerPRS"<<endl;
    for(int i=0;i!=DiffPRSCounter;i++)
    {
        ResSavedPerPRS[i] = 0;
        cout<<ResSavedPerPRS[i]<<"\t";
    }
    cout<<endl;
    if(world_rank > 0)
        sleep(0.01);
    if(world_rank == 0 && world_size > 1)
    {
        cout<<"Rank "<<world_rank<<" starting!"<<endl;
        int NFreeNodes = getNFreeNodes(NodeBusy,world_size-1);
        int NStartedTasks = getNStartedTasks(TaskFinished,NTasks);
        int NFinishedTasks = getNFinishedTasks(TaskFinished,NTasks);
        while((NStartedTasks < NTasks || NFinishedTasks < NTasks)  && world_size > 1)
        {
            NFreeNodes = getNFreeNodes(NodeBusy,world_size-1);
            int TaskToStart = -1;
            for(int i=0;i!=NTasks;i++)
            {
                if(TaskFinished[i] == 0)
                {
                    TaskToStart = i;
                    break;
                }
            }
            if(NFreeNodes > 0 && TaskToStart != -1)
            {
                int NodeToUse = -1;
                for(int i=0;i!=world_size-1;i++)
                {
                    if(NodeBusy[i] == 0)
                    {
                        NodeToUse = i;
                        break;
                    }
                }
                NodeTask[NodeToUse] = TaskToStart;
                TaskFinished[TaskToStart] = 1;
                MPI_Send(&PRS[TaskToStart],PRSsize,MPI_BYTE,NodeToUse+1,0,MPI_COMM_WORLD);
                cout<<"sent task "<<TaskToStart<<" to "<<NodeToUse+1<<endl;
                NodeBusy[NodeToUse] = 1;
            }
            else
            {
                cout<<world_rank<<" Receiving result"<<endl;
                Result ReceivedRes;
                MPI_Recv(&ReceivedRes,ResultSize,MPI_BYTE,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                AllResults[NodeTask[ReceivedRes.Node]].Copy(ReceivedRes,ResTsize1,ResTsize2);
                cout<<world_rank<<" received from \t"<<ReceivedRes.Node<<endl;
                NodeBusy[ReceivedRes.Node] = 0;
                TaskFinished[NodeTask[ReceivedRes.Node]] = 2;

                for(int i=0;i!=DiffPRSCounter;i++)
                {
                    int totalFinval = 0;
                    for(int j=0;j!=TotalNRuns;j++)
                    {
                        totalFinval += TaskFinished[i*TotalNRuns+j];
                    }
                    if(totalFinval == TotalNRuns*2 && ResSavedPerPRS[i] == 0)
                    {
                        cout<<"Saving to:  "<<FilePath[NodeTask[ReceivedRes.Node]]<<endl;
                        cout<<"Saving to:  "<<FilePathC[NodeTask[ReceivedRes.Node]]<<endl;
                        cout<<"Saving to:  "<<FilePathC2[NodeTask[ReceivedRes.Node]]<<endl;
                        cout<<"Saving to:  "<<FilePathShort[NodeTask[ReceivedRes.Node]]<<endl;
                        cout<<"Saving to:  "<<FilePathShortC[NodeTask[ReceivedRes.Node]]<<endl;
                        ResSavedPerPRS[i] = 1;
                        ofstream fout(FilePath[NodeTask[ReceivedRes.Node]]);
                        ofstream foutC(FilePathC[NodeTask[ReceivedRes.Node]]);
                        ofstream foutC2(FilePathC2[NodeTask[ReceivedRes.Node]]);
						ofstream foutShort(FilePathShort[NodeTask[ReceivedRes.Node]]);
						ofstream foutShortC(FilePathShortC[NodeTask[ReceivedRes.Node]]);
						fout.precision(18);
						foutC.precision(18);
						foutC2.precision(18);
						foutShort.precision(18);
						foutShortC.precision(18);
                        for(int RunN=0;RunN!=TotalNRuns;RunN++)
                        {
                            for(int func_num=0;func_num!=ResTsize1;func_num++)
                            {
                                for(int k=0;k!=ResTsize2;k++)
                                {
                                    fout<<AllResults[i*TotalNRuns+RunN].ResultTable1[func_num][k]<<"\t";
                                    for(int L=0;L!=maxGconstr;L++)
                                        foutC<<AllResults[i*TotalNRuns+RunN].ResultTableG1[func_num][k][L]<<"\t";
                                    for(int L=0;L!=maxHconstr;L++)
                                        foutC<<AllResults[i*TotalNRuns+RunN].ResultTableH1[func_num][k][L]<<"\t";
                                    for(int L=0;L!=global_gn[func_num];L++)
                                        foutC2<<AllResults[i*TotalNRuns+RunN].ResultTableG1[func_num][k][L]<<"\t";
                                    for(int L=0;L!=global_hn[func_num];L++)
                                        foutC2<<AllResults[i*TotalNRuns+RunN].ResultTableH1[func_num][k][L]<<"\t";
                                }
                                fout<<endl;
                                foutC<<endl;
                                foutC2<<endl;

								foutShort<<AllResults[i*TotalNRuns+RunN].ResultTable1[func_num][ResTsize2-2]<<"\t";
								double totalViolS = 0;
								for(int L=0;L!=global_gn[func_num];L++)
								{
									if(AllResults[i*TotalNRuns+RunN].ResultTableG1[func_num][ResTsize2-2][L] > 0)
										totalViolS += AllResults[i*TotalNRuns+RunN].ResultTableG1[func_num][ResTsize2-2][L];
								}
								for(int L=0;L!=global_hn[func_num];L++)
								{
									if(fabs(AllResults[i*TotalNRuns+RunN].ResultTableG1[func_num][ResTsize2-2][L]) > 0.0001)
										totalViolS += fabs(AllResults[i*TotalNRuns+RunN].ResultTableH1[func_num][ResTsize2-2][L]);
								}
								foutShortC<<totalViolS<<"\t";
                            }
							foutShort<<endl;
							foutShortC<<endl;
                        }
                        fout.close();
                    }
                }
            }
            string task_stat;
            task_stat = "";
            for(int i=0;i!=world_size-1;i++)
            {
                sprintf(buffer,"%d",NodeBusy[i]);
                task_stat += buffer;
            }
            cout<<"NodeBusy: "<<task_stat<<endl;
            cout<<"Total NTasks: "<<NTasks<<endl;

            task_stat = "";
            for(int i=0;i!=world_size-1;i++)
            {
                sprintf(buffer,"%d ",NodeTask[i]);
                task_stat += buffer;
            }
            cout<<"NodeTask: "<<task_stat<<endl;

            task_stat = "";
            for(int i=0;i!=NTasks;i++)
            {
                sprintf(buffer,"%d",TaskFinished[i]);
                task_stat += buffer;
            }
            NStartedTasks = getNStartedTasks(TaskFinished,NTasks);
            NFinishedTasks = getNFinishedTasks(TaskFinished,NTasks);
            cout<<"NFINISHED "<<NFinishedTasks<<endl;
        }
        cout<<world_rank<<" sending Finish"<<endl;
        for(int i=1;i!=world_size;i++)
        {
            Params PRSFinish;
            PRSFinish.Type = -1;
            MPI_Send(&PRSFinish,PRSsize,MPI_BYTE,i,0,MPI_COMM_WORLD);
        }
    }
    else
    {
        int CurTask = 0;
		int NFinishedTasks = 0;
        while(NFinishedTasks < NTasks)
        {
            Params CurPRS;
            if(world_size > 1)
            {
                MPI_Recv(&CurPRS,PRSsize,MPI_BYTE,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                cout<<world_rank<<" received"<<endl;
                if(CurPRS.Type == -1)
                {
                    cout<<world_rank<<" Finishing!"<<endl;
                    break;
                }
            }
            else
            {
                if(CurTask > NTasks)
                    break;
                CurPRS = PRS[CurTask];
            }
            Result ResToSend;
            ResToSend.Node=world_rank-1;
            ResToSend.Task=0;

            //////////////////////////////////////////////////////////////////////

            int maxNFunc = 57;
            unsigned algtime_0=clock(),algtime_1;
            for(int func_num = 1; func_num < maxNFunc+1; func_num++)
            {
                GNVars = global_D[func_num-1];
                //int NumG = global_gn[func_num-1];
                //int NumH = global_hn[func_num-1];
                if(GNVars <= 10)
                    MaxFEval = 10000;
                else if(GNVars <= 30)
                    MaxFEval = 20000;
                else if(GNVars <= 50)
                    MaxFEval = 40000;
                else if(GNVars <= 150)
                    MaxFEval = 80000;
                else
                    MaxFEval = 100000;
                ResultsArray[func_num-1][ResTsize2-1] = MaxFEval;
                //getOptimum(func_num);
                globalbestinit = false;
                LastFEcount = 0;
                NFEval = 0;
                Cvalglobal = CurPRS.CvalLoop;
                EpsIndexglobal = CurPRS.EpsIndexLoop;
                GRProbGlobal = CurPRS.GRProbLoop;
                ConfigGlobal = CurPRS.ConfigLoop;
                SelPresGlobal = CurPRS.SelPresLoop;
                CRSGlobal = CurPRS.CRSLoop;
                int NewPopSize = CurPRS.PopSizeLoop*10;
                Optimizer OptZ;
                OptZ.Initialize(NewPopSize, GNVars, func_num);
                OptZ.MainCycle();
                OptZ.Clean();
                for(int j=0;j!=ResTsize2;j++)
                {
                    ResToSend.ResultTable1[func_num-1][j] = ResultsArray[func_num-1][j];
                    for(int k=0;k!=maxGconstr;k++)
                        ResToSend.ResultTableG1[func_num-1][j][k] = ResultsArrayG[func_num-1][j][k];
                    for(int k=0;k!=maxHconstr;k++)
                        ResToSend.ResultTableH1[func_num-1][j][k] = ResultsArrayH[func_num-1][j][k];
                }
            }
            algtime_1=clock();
            sprintf(buffer, "time_%d.txt", world_rank);
            //ofstream fout_time(buffer);
            algtime_1 = clock() - algtime_0;
            cout<<algtime_1<<endl;
            //fout_time<<GRProbGlobal<<"\t"<<double(algtime_1)/double(CLOCKS_PER_SEC)<<endl;
            if(world_size > 1)
            {
                cout<<world_rank<<" sending to 0 result "<<endl;
                MPI_Send(&ResToSend,ResultSize,MPI_BYTE,0,0,MPI_COMM_WORLD);
                sleep(0.1);
            }
            else
            {
                TaskFinished[CurTask] = 2;
                for(int i=0;i!=DiffPRSCounter;i++)
                {
                    int totalFinval = 0;
                    for(int j=0;j!=TotalNRuns;j++)
                    {
                        totalFinval += TaskFinished[i*TotalNRuns+j];
                    }
                    cout<<"totalFinVal "<<totalFinval<<endl;
                    if(totalFinval == TotalNRuns*2 && ResSavedPerPRS[i] == 0)
                    {
                        cout<<"Saving to:  "<<FilePath[CurTask]<<endl;
                        cout<<"Saving to:  "<<FilePathC[CurTask]<<endl;
                        cout<<"Saving to:  "<<FilePathC2[CurTask]<<endl;
                        cout<<"Saving to:  "<<FilePathShort[CurTask]<<endl;
                        cout<<"Saving to:  "<<FilePathShortC[CurTask]<<endl;
                        ResSavedPerPRS[i] = 1;
                        ofstream fout(FilePath[CurTask]);
                        ofstream foutC(FilePathC[CurTask]);
                        ofstream foutC2(FilePathC2[CurTask]);
						ofstream foutShort(FilePathShort[CurTask]);
						ofstream foutShortC(FilePathShortC[CurTask]);
						fout.precision(18);
						foutC.precision(18);
						foutC2.precision(18);
						foutShort.precision(18);
						foutShortC.precision(18);
                        for(int func_num=0;func_num!=ResTsize1;func_num++)
                        {
                            for(int k=0;k!=ResTsize2;k++)
                            {
                                fout<<AllResults[i*TotalNRuns+0].ResultTable1[func_num][k]<<"\t";
                                for(int L=0;L!=maxGconstr;L++)
                                    foutC<<AllResults[i*TotalNRuns+0].ResultTableG1[func_num][k][L]<<"\t";
                                for(int L=0;L!=maxHconstr;L++)
                                    foutC<<AllResults[i*TotalNRuns+0].ResultTableH1[func_num][k][L]<<"\t";
                                for(int L=0;L!=global_gn[func_num];L++)
                                    foutC<<AllResults[i*TotalNRuns+0].ResultTableG1[func_num][k][L]<<"\t";
                                for(int L=0;L!=global_hn[func_num];L++)
                                    foutC<<AllResults[i*TotalNRuns+0].ResultTableH1[func_num][k][L]<<"\t";
                            }
                            fout<<endl;
                            foutC<<endl;
                            foutC2<<endl;

                            foutShort<<AllResults[i*TotalNRuns+0].ResultTable1[func_num][ResTsize2-2]<<"\t";
                            double totalViolS = 0;
                            for(int L=0;L!=global_gn[func_num];L++)
                            {
                                if(AllResults[i*TotalNRuns+0].ResultTableG1[func_num][ResTsize2-2][L] > 0)
                                    totalViolS += AllResults[i*TotalNRuns+0].ResultTableG1[func_num][ResTsize2-2][L];
                            }
                            for(int L=0;L!=global_hn[func_num];L++)
                            {
                                if(fabs(AllResults[i*TotalNRuns+0].ResultTableG1[func_num][ResTsize2-2][L]) > 0.0001)
                                    totalViolS += fabs(AllResults[i*TotalNRuns+0].ResultTableH1[func_num][ResTsize2-2][L]);
                            }
                            foutShortC<<totalViolS<<"\t";
                        }
                        foutShort<<endl;
                        foutShortC<<endl;
                        fout.close();
                        foutC.close();
                        foutC2.close();
                        foutShort.close();
                        foutShortC.close();
                    }
                }
                CurTask++;
            }
			NFinishedTasks = getNFinishedTasks(TaskFinished,NTasks);	
        }
    }

    delete NodeBusy;
    delete NodeTask;
    if(world_rank == 0)
        cout<<"Rank "<<world_rank<<"\ton\t"<<processor_name<<"\t Finished at"<<"\t"<<currentDateTime()<<"\n";
    MPI_Finalize();

    t1g=clock()-t0g;
    double T0g = t1g/double(CLOCKS_PER_SEC);
    if(world_rank == 0)
    {
        cout << "Time spent: " << T0g << endl;
    }
	return 0;
}

