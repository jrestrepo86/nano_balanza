
%Limpiamos todas las variables del Workspace
clear all;
close all;
clc;

%parametros generales de la simulación
n_points=5000;%número de puntos
f_ini=4.8e6;%frecuencia inicial
f_final=5.2e6;%frecuencia final
%Definición de los parámetros del resonador (5MHz)
Lm=64.1394e-3;
Cm=16.0371e-15;
Rm=11.42;
C0=43.3903e-12;
Lmass1=0.1e-3;
Lmass2=0.2e-3;
Lmass3=0.3e-3;
coupling=linspace(0.001,0.005,5);
%Lm=unifrnd(0, 30 ,4 , 10000);

%figure(2);

for i = 1:5
%Actualizamos el valor de acoplamiento Lk
Lk=coupling(i)*Lm;

parameters=[Lm,Cm,Rm,Lk,C0,Lmass1,Lmass2,Lmass3];

fsim=linspace(f_ini,f_final,n_points);
[Zsim] = simulate_Y_4resonators_singleLcoupling_model(parameters,fsim);



Ysim=1./Zsim;
Gsim=real(Ysim);
Bsim=imag(Ysim);

figure(1);plot(fsim,20*log10(abs(Zsim)));
hold on;grid on;
figure(2);plot(fsim,real(Zsim));
hold on;grid on;
figure(3);plot(fsim,Gsim);
hold on;grid on;



end
