System Wide Audio Equalizer and Effects Processor
=====================================================

Table of Contents
-----------------

1. [Overview](#overview)
2. [Audio Processing Algorithms](#audio-processing-algorithms)
	* [Equalization](#equalization)
	* [Reverb](#reverb)
	* [Distortion](#distortion)
	* [Delay](#delay)
	* [Compression](#compression)
3. [Implementation Details](#implementation-details)
4. [Technical Requirements](#technical-requirements)
5. [Future Work](#future-work)
6. [Contributing](#contributing)
7. [License](#license)

### Overview

This project is a system-wide audio equalizer and effects processor that utilizes various audio processing algorithms to enhance and modify audio signals in real-time.

### Audio Processing Algorithms

The project employs a range of audio processing algorithms to achieve its functionality.

#### Equalization

The equalization algorithm used in this project is a 10-band parametric equalizer. The equalizer allows users to adjust the gain of specific frequency bands, enabling precise tone shaping and correction.

* **Algorithm:** Parametric equalization using a second-order IIR filter
* **Implementation:** The equalizer is implemented using a cascade of second-order IIR filters, each responsible for a specific frequency band.

#### Reverb

The reverb algorithm used in this project is based on the Schroeder-Moorer reverb algorithm. This algorithm simulates the natural reverberation of a physical space by generating a series of delayed and attenuated copies of the input signal.

* **Algorithm:** Schroeder-Moorer reverb algorithm using a combination of comb filters and all-pass filters
* **Implementation:** The reverb is implemented using a network of comb filters and all-pass filters, which are used to generate the delayed and attenuated copies of the input signal.

#### Distortion

The distortion algorithm used in this project is a non-linear distortion algorithm that applies a hyperbolic tangent function to the input signal. This type of distortion is commonly used in audio processing to create a "warm" or "overdriven" sound.

* **Algorithm:** Non-linear distortion using a hyperbolic tangent function
* **Implementation:** The distortion is implemented using a simple non-linear function that applies the hyperbolic tangent to the input signal.

#### Delay

The delay algorithm used in this project is a simple delay line that stores a copy of the input signal and plays it back after a specified delay time.

* **Algorithm:** Simple delay line using a circular buffer
* **Implementation:** The delay is implemented using a circular buffer that stores the input signal and plays it back after a specified delay time.

#### Compression

The compression algorithm used in this project is a dynamic range compressor that reduces the dynamic range of the input signal. The compressor uses an envelope follower to track the amplitude of the input signal and applies gain reduction accordingly.

* **Algorithm:** Dynamic range compression using an envelope follower and gain reduction
* **Implementation:** The compressor is implemented using an envelope follower that tracks the amplitude of the input signal and applies gain reduction using a simple gain reduction algorithm.

### Future Work

Future work on this project may include:

* Implementing additional audio effects, such as chorus or flanger
* Improving the user interface and user experience
* Optimizing the audio processing algorithms for performance

### License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## References
https://stackoverflow.com/questions/77611711/implementation-differences-between-feedforward-and-backward-comb-filters
https://www.reddit.com/r/DSP/comments/1baxiyo/reverb_algorithm/
https://github.com/SergeiGorskiy/IFMO_Reverb
