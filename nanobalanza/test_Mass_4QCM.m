%Limpiamos todas las variables del Workspace
%clear all;

%parametros generales de la simulación
n_points=50000;%número de puntos
f_ini=4.9e6;%frecuencia inicial
f_final=5e6;%frecuencia final
%Definición de los parámetros del resonador (5MHz)
Lm=64.1394e-3;
Cm=16.0371e-15;
Rm=11.42;
C0=43.3903e-12;
Lk=0.002*Lm;

mass=linspace(0,0.5e-3,5);

figure(3);

for i = 1:5
%Actualizamos el valor de L debido a masa (Lmass)
Lmass=mass(i);


parameters=[Lm,Cm,Rm,Lk,Lmass,C0,Lmass1,Lmass2];

fsim=linspace(f_ini,f_final,n_points);
[Zsim] = simulate_Y_4resonators_singleLcoupling_model(parameters,fsim);



Ysim=1./Zsim;
Gsim=real(Ysim);
Bsim=imag(Ysim);

figure(1);plot(fsim,20*log10(abs(Zsim)));
hold on;
figure(2);plot(fsim,real(Zsim));
hold on;
figure(3);plot(fsim,Gsim);
hold on;
end
