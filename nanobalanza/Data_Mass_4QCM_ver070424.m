%Limpiamos todas las variables del Workspace
clear all;
close all;
clc;

%parametros generales de la simulación
n_points=30000;%número de puntos
f_ini=4.965e6;%frecuencia inicial
f_final=4.995e6;%frecuencia final
%Definición de los parámetros del resonador (5MHz)
Lm=64.1394e-3;
Cm=16.0371e-15;
Rm=11.42;
C0=43.3903e-12;
Lk=0.005*Lm;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Suponiendo que las cuatro masas pueden variar, aquí habría que 
% generar una matriz de valores aleatorios de las masas. Esta matriz
% debería contener elementos con distribución uniforme entre el mínimo 
% y el máximo valor de interés (podemos llamarlo Masa_min y Masa_max). 
% El tamaño de la matriz será de 3 filas y N columnas, donde N es la 
% cantidad de datos a simular. Por ejemplo N=10000. Entonces:
Masa_min=64.1394e-9;
Masa_max=30*64.1394e-9;
N=1000;


Lmass=unifrnd(Masa_min, Masa_max, 3, N);

% No sé qué significa el parámetro coupling (es la inductacia generada por el acoplamiento entre sensores), si tenemos que barrer esos
% 5 valores o se puede fijar. El próximo paso es recorrer los N valores de 
% las masas con un for:

for n=1:N
 Lmass1=Lmass(1,n);
 Lmass2=Lmass(2,n);
 Lmass3=Lmass(3,n);

% aquí va el resto del código de simulación para hallar el espectro
        % for i = 1:5
        % %Actualizamos el valor de L debido a masa (Lmass)
        % Lmass1=mass(i);
        % 
        parameters=[Lm,Cm,Rm,C0,Lk,Lmass1,Lmass2,Lmass3];
        % 
        % 
        fsim=linspace(f_ini,f_final,n_points);
        [Zsim] = New_simulate_Y_4resonators_singleLcoupling_model(parameters,fsim);
        % 
        % 
        Ysim=1./Zsim;
        Gsim=real(Ysim);
        Bsim=imag(Ysim);
        % 
        figure(1);plot(fsim,20*log10(abs(Zsim)));
        hold on;grid on;
        % figure(2);plot(fsim,real(Zsim));
        % hold on;
        % figure(3);plot(fsim,Gsim);
        % hold on;

 %    ......
 % y guardas el espectro en el formato de interés.  
 % Tenés que discutir los valores de masa mínimo y máximo con tus otros directores,
 % y qué hacer con "coupling" y qué espectro guardar.
 % guardás Gsim (parte real) dentro del "for" en una matriz, supongamos que se llama G así:
    G(:,n)=Gsim';
end
















