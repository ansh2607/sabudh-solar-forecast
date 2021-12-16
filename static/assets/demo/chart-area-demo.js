// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#292b2c';

var csv_file = "static/csv/IN0666B.csv"
function load_csv(parm1) {
str1 = "static/csv/",
csv_file1 = str1 + district[String(parm1)];
} 
d3.csv(csv_file)
.then(makeChart);
// Area Chart Example
function makeChart(weather) {
    var weatherdate = weather.map(function(d) {return d.date});
    var temp = weather.map(function(d) {return d.temp});
    var percp = weather.map(function(d) {return d.prcp});
    var maxspeed = weather.map(function(d) {return d.mxspd});
    var ctx = document.getElementById("temp");
    var myLineChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: weatherdate,
        datasets: [{
          label: "Sessions",
          lineTension: 0.3,
          backgroundColor: "rgba(2,117,216,0.2)",
          borderColor: "rgba(2,117,216,1)",
          pointRadius: 5,
          pointBackgroundColor: "rgba(2,117,216,1)",
          pointBorderColor: "rgba(255,255,255,0.8)",
          pointHoverRadius: 5,
          pointHoverBackgroundColor: "rgba(2,117,216,1)",
          pointHitRadius: 50,
          pointBorderWidth: 2,
          data: temp
        }],
      },
      options: {
        scales: {
          xAxes: [{
            time: {
              unit: 'date'
            },
            gridLines: {
              display: false
            },
            ticks: {
              maxTicksLimit: 20
            }
          }],
          yAxes: [{
            ticks: {
              min: 0,
              max: 40,
              maxTicksLimit: 10
            },
            gridLines: {
              color: "rgba(0, 0, 0, .125)",
            }
          }],
        },
        legend: {
          display: false
        }
      }
    });
    var ctx = document.getElementById("maxspeed");
    var myLineChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: weatherdate,
        datasets: [{
          label: "Sessions",
          lineTension: 0.3,
          backgroundColor: "rgba(2,117,216,0.2)",
          borderColor: "rgba(2,117,216,1)",
          pointRadius: 5,
          pointBackgroundColor: "rgba(2,117,216,1)",
          pointBorderColor: "rgba(255,255,255,0.8)",
          pointHoverRadius: 5,
          pointHoverBackgroundColor: "rgba(2,117,216,1)",
          pointHitRadius: 50,
          pointBorderWidth: 2,
          data: maxspeed
        }],
      },
      options: {
        scales: {
          xAxes: [{
            time: {
              unit: 'date'
            },
            gridLines: {
              display: false
            },
            ticks: {
              maxTicksLimit: 20
            }
          }],
          yAxes: [{
            ticks: {
              min: 0,
              max: 10,
              maxTicksLimit: 10
            },
            gridLines: {
              color: "rgba(0, 0, 0, .125)",
            }
          }],
        },
        legend: {
          display: false
        }
      }
    });
    var ctx = document.getElementById("precip");
    var myLineChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: weatherdate,
        datasets: [{
          label: "Sessions",
          lineTension: 0.3,
          backgroundColor: "rgba(2,117,216,0.2)",
          borderColor: "rgba(2,117,216,1)",
          pointRadius: 5,
          pointBackgroundColor: "rgba(2,117,216,1)",
          pointBorderColor: "rgba(255,255,255,0.8)",
          pointHoverRadius: 5,
          pointHoverBackgroundColor: "rgba(2,117,216,1)",
          pointHitRadius: 50,
          pointBorderWidth: 2,
          data: percp
        }],
      },
      options: {
        scales: {
          xAxes: [{
            time: {
              unit: 'date'
            },
            gridLines: {
              display: false
            },
            ticks: {
              maxTicksLimit: 20
            }
          }],
          yAxes: [{
            ticks: {
              min: 0,
              max: 70,
              maxTicksLimit: 10
            },
            gridLines: {
              color: "rgba(0, 0, 0, .125)",
            }
          }],
        },
        legend: {
          display: false
        }
      }
    });
  }