#pragma once

#include <QtWidgets/QMainWindow>
#include <Qpainter>
#include <QtCharts>
#include <QChartView>
#include <QPieSeries>
#include <QPieSlice>
#include "ui_TSPGUItest.h"
#include "graph.h"

class TSPGUItest : public QMainWindow
{
    Q_OBJECT

public:
    TSPGUItest(QWidget *parent = nullptr);
    ~TSPGUItest();
    graph my_graph;
    QLineSeries* series1;
    QScatterSeries* series;
    QScatterSeries* src;
    QChart* chart;

    // virtual void paintEvent(QPaintEvent *event, QString);

private:
    Ui::TSPGUItestClass ui;

private slots:
    void generate_graph();
    void compute();
};
