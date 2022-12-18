#include "TSPGUItest.h"
#include "TSP.cuh"

TSPGUItest::TSPGUItest(QWidget *parent) : 
	QMainWindow(parent)
{
    ui.setupUi(this);
    connect(ui.pushButton_generate_graph, SIGNAL(clicked()), this, SLOT(generate_graph()));
	connect(ui.pushButton_compute, SIGNAL(clicked()), this, SLOT(compute()));

	series1 = new QLineSeries();
	series = new QScatterSeries();
	src = new QScatterSeries();

	chart = new QChart();

	chart->addSeries(series1);
	chart->addSeries(series);
	chart->addSeries(src);
	
	chart->setTitle("TSP solution");

	chart->createDefaultAxes();
	
	chart->legend()->hide();

	QChartView* chartview = new QChartView(chart);
	chartview->resize(772, 701);
	chartview->setRenderHint(QPainter::Antialiasing);
	chartview->setParent(ui.horizontalFrame);
	
}

TSPGUItest::~TSPGUItest() {
	free(src);
	free(series1);
	free(series);
	free(chart);
}


void TSPGUItest::generate_graph() {
    my_graph.cities = ui.spinBox_number_of_cities->value();
    qInfo() << "generate graph with: " << my_graph.cities << " cities";
	
	src->clear();
	series1->clear();
	series->clear();

	int m = my_graph.cities;

	unsigned long long solutions_number = factorial(m - 1);
	// -------------------------------------------------- GRAF -----------------------------------------------

	// macierz położeń - m x m, w przestrzeni kartezjańskiej 2d ale zapisana w formie tablicy 1d, zamiast 2d
	// kazdy punkt (miasto) ma współrzędne x oraz y

	//my_graph.location = (int*)malloc(static_cast<size_t>(static_cast<size_t>(2 * m) * sizeof(int)));


	if (my_graph.location == NULL) {
		qInfo() << "zjebany wskaxnik";
	}

	// losowanie unikatowych punktów w przestrzeni
	srand(time(NULL));

	bool point_allready_exist = false;
	int x, y;

	for (int i = 0; i < 2 * m; i += 2) {
		// losowanie punktu, którego jeszcze nie było
		point_allready_exist = false;

		// realizuj tak długo, aż wylosowany punkt będzie inny od istniejących
		do {
			// przyjmujemy, że punkt jest unikatowy
			point_allready_exist = false;

			// losujemy współrzędne
			x = rand() % m;
			y = rand() % m;

			// sprawdzenie, czy nie istnieje już punkt o tych współrzędnych
			for (int j = 0; j <= i; j += 2) {
				// sprawdzenie współrzędnej X
				if (x == my_graph.location[i - j]) {
					// sprawdzenie współrzędnej Y
					if (y == my_graph.location[i - j + 1]) {
						point_allready_exist = true;
					}
				}
			}

		} while (point_allready_exist);

		my_graph.location[i] = x;
		my_graph.location[i + 1] = y;
		qInfo() << "x: " << my_graph.location[i] << "   y: " << my_graph.location[i+1];

		series->append(x, y);
	}
	

	series->setMarkerShape(QScatterSeries::MarkerShapeCircle);
	series->setMarkerSize(15.0);

	chart->axes(Qt::Horizontal).first()->setRange(-1, my_graph.cities);
	chart->axes(Qt::Vertical).first()->setRange(-1, my_graph.cities);

	chart->update();

}


void TSPGUItest::compute() {
	my_graph.start_city = ui.spinBox_start_city->value();
	qInfo() << "computing with start city: " << my_graph.start_city;

	series1->clear();
	src->clear();

	src->append(my_graph.location[2 * my_graph.start_city], my_graph.location[2 * my_graph.start_city + 1]);

	fromCUDA data;

	data = Wrapper::wrapper(my_graph.cities, my_graph.location, my_graph.start_city);
	

	qInfo() << "info : " << data.e;
	qInfo() << data.GPU_time << "us";

	unsigned long target_time_CPU = 0;
	unsigned long target_time_GPU = 0;

	if (data.CPU_time > 1000000 && data.GPU_time > 1000000) {
		target_time_GPU = data.GPU_time / 1000000;
		target_time_CPU = data.CPU_time / 1000000;
		ui.lineEdit_GPU_time->setText(QString("%1 s").arg(target_time_GPU));
		ui.lineEdit_CPU_time->setText(QString("%1 s").arg(target_time_CPU));
	}
	else if (data.CPU_time > 1000 && data.GPU_time > 1000) {
		target_time_GPU = data.GPU_time / 1000;
		target_time_CPU = data.CPU_time / 1000;
		ui.lineEdit_GPU_time->setText(QString("%1 ms").arg(target_time_GPU));
		ui.lineEdit_CPU_time->setText(QString("%1 ms").arg(target_time_CPU));
	}
	else {
		ui.lineEdit_GPU_time->setText(QString("%1 us").arg(data.GPU_time));
		ui.lineEdit_CPU_time->setText(QString("%1 us").arg(data.CPU_time));
	}

	// ui.lineEdit_GPU_time->setText(QString("%1 us").arg(data.GPU_time));
	// ui.lineEdit_CPU_time->setText(QString("%1 us").arg(data.CPU_time));
	
	double speedup = static_cast<double>(data.CPU_time) / static_cast<double>(data.GPU_time);
	
	ui.lineEdit_SpeedUp->setText(QString::number(speedup, 'f', 3));
	
	for (int i = 0; i < my_graph.cities + 1; i++) {
		qInfo() << data.cites[i];
		series1->append(my_graph.location[2 * data.cites[i]], my_graph.location[2 * data.cites[i] + 1]);
	}

	chart->update();
}
