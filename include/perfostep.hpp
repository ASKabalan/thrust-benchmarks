#include <chrono>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
#include <thrust/transform.h>
#include <iostream>


using namespace std::chrono;

class Perfostep {
private:
  time_point<system_clock> m_StartTime;
  time_point<system_clock> m_EndTime;
  bool m_bRunning = false;

public:
  void start() {
    m_StartTime = system_clock::now();
    m_bRunning = true;
  }

  double stop() {
    m_EndTime = system_clock::now();
    m_bRunning = false;
        return elapsedMilliseconds();
  }

  double elapsedMilliseconds() {
    time_point<system_clock> endTime;

    if (m_bRunning) {
      endTime = system_clock::now();
    } else {
      endTime = m_EndTime;
    }

    return duration_cast<milliseconds>(endTime - m_StartTime).count();
  }

  double elapsedSeconds() { return elapsedMilliseconds(); }

  inline void report(const char *msg) {
    std::cout << msg << " is done, elapsed time: " << elapsedMilliseconds()
              << "ms " << std::endl;
  }
  inline void report(const double &elapsed_seconds, const char *msg) {
    std::cout << msg << " is done, elapsed time: " << elapsed_seconds
              << "ms "<<  std::endl;
  }
};
