# USTransmission
This repository cointains a Python simulator for medical ultrasound linear probe beam pattern computation.
In detail, the code consists into two different approaches of beam pattern computation as is different for wide-band and narrow-band transmission settings.
Furthermore, it makes use of some approximations and exploits geometrical simmetries as described in [1].
The simulation consists in computing or loading an approximation of the impulse responses maps for a choice of probe and a field of research [2], and computing the beam pattern 
as the power of the signal in time that crosses a point of the field having fixed a number of active element, a pulse emitted and a set of delays.



[1] Cit archivio paper teorico nostro ?

[2] J. A. Jensen. A new calculation procedure for spatial impulse responses in ultrasound. The Journal of the Acoustical Society of America, 105(6):3266–3274, 06 1999
