#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define NHID 3
#define NIO 8
using namespace std;

// NIO input, NHID hidden, NIO output
double w_i2h[NHID][NIO], w_h2o[NIO][NHID], e_h[NHID], e_o[NIO], b_h[NHID], b_o[NIO];

// const
double rate = 0.05;

double randTiny()
{
    // result is in 0.01~0.1
    
    int t = rand() % 1000;
    int bi = rand() % 2;
    if (bi)
        t = -t;
    return t / 10000.0;
    //return 0.1;
}

void annInit()
{
    srand(time(0));
    for (int i = 0; i < NHID; i++)
        for (int j = 0; j < NIO; j++)
        {
            w_i2h[i][j] = randTiny();
            w_h2o[j][i] = randTiny();
        }
    for (int i = 0; i < NHID; i++) b_h[i] = randTiny();
    for (int j = 0; j < NIO; j++) b_o[j] = randTiny();
}

double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

double backPropagation(int x[NIO]) // in this problem, we have x == t +- 0.1
{
    double a[NHID], o[NIO]; // values of hidden units and output units
    double f_h2o[NIO][NHID], f_i2h[NHID][NIO]; // flow in network
    double t[NIO];

    for (int i = 0; i < NIO; i++)
        if (x[i])
            t[i] = 0.9;
        else
            t[i] = 0.1;

    // forward propagation
    for (int i = 0; i < NHID; i++)
    {
        a[i] = b_h[i];
        for (int j = 0; j < NIO; j++)
        {
            f_i2h[i][j] = (w_i2h[i][j] * x[j]);
            a[i] += f_i2h[i][j];
        }
        a[i] = sigmoid(a[i]);
    }
    for (int i = 0; i < NIO; i++)
    {
        o[i] = b_o[i];
        for (int j = 0; j < NHID; j++)
        {
            f_h2o[i][j] = (w_h2o[i][j] * a[j]);
            o[i] += f_h2o[i][j];
        }
        o[i] = sigmoid(o[i]);
    }

    // error count
    double rst = 0;
    for (int i = 0; i < NIO; i++)
        rst += (o[i] - t[i]) * (o[i] - t[i]);

    // backward propagation
    for (int i = 0; i < NIO; i++)
        e_o[i] = o[i] * (1 - o[i]) * (t[i] - o[i]);
    for (int i = 0; i < NHID; i++)
    {
        e_h[i] = 0;
        for (int j = 0; j < NIO; j++)
            e_h[i] += a[i] * (1 - a[i]) * w_h2o[j][i] * e_o[j];
    }

    // weights refresh
    for (int i = 0; i < NIO; i++)
    {
        for (int j = 0; j < NHID; j++)
            w_h2o[i][j] += rate * e_o[i] * f_h2o[i][j];
        b_o[i] += rate * e_o[i];
    }
    for (int i = 0; i < NHID; i++)
    {
        for (int j = 0; j < NIO; j++)
            w_i2h[i][j] += rate * e_h[i] * f_i2h[i][j];
        b_h[i] += rate * e_h[i];
    }
    
    return rst;
}

void annRun(int x[NIO])
{
    double a[NHID], o[NIO];
    for (int i = 0; i < NHID; i++)
    {
        a[i] = b_h[i];
        for (int j = 0; j < NIO; j++)
            a[i] += (w_i2h[i][j] * x[j]);
        a[i] = sigmoid(a[i]);
    }
    for (int i = 0; i < NIO; i++)
    {
        o[i] = b_o[i];
        for (int j = 0; j < NHID; j++)
            o[i] += (w_h2o[i][j] * a[j]);
        o[i] = sigmoid(o[i]);
    }
    for (int i = 0; i < NHID; i++)
        printf("%.3lf ", a[i]);
    printf("\n");
    for (int i = 0; i < NIO; i++)
        if (x[i] == 1)
            printf("%.3lf\n", o[i]);
    /*
    for (int i = 0; i < NIO; i++)
        cout << o[i] << ' ';
    cout << endl;
    */
}

int main()
{
    annInit();
    int time;
    cin >> time;
    for (int idTime = 0; idTime < time; idTime++)
    {
        int x[NIO] = {0};
        int i = rand() % 8;
        x[i] = 1;
        backPropagation(x);
        x[i] = 0;
    }
    int x[NIO] = {0};
    for (int i = 0; i < NIO; i++)
    {
        cout << i + 1 << " =================== " << endl;
        x[i] = 1;
        annRun(x);
        x[i] = 0;
    }
    return 0;
}
