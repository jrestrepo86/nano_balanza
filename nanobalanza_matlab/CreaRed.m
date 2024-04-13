
cd /media/Datos/nanobalanza
load datos.mat;
Normaliza;

x = Fe_Norm;

t = M_Norm;
clear Fe_Norm M_Norm
 
trainFcn = 'trainlm';  %  

 
hiddenLayerSize1 = 50;
hiddenLayerSize2 = 20;
hiddenLayerSize3 = 6;

%hiddenLayerSize3 = 10;hiddenLayerSize3

net = fitnet([hiddenLayerSize1 hiddenLayerSize2  ],trainFcn);


net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};


net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 85/100;
%net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'mse';  % Mean Squared Error

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotregression', 'plotfit'};
net.trainParam.epochs=20000;
net.trainParam.max_fail=50;
% Train the Network
[net,tr] = train(net,x,t,'useGPU','yes');

% Test the Network
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y)

% Recalculate Training, Validation and Test Performance
trainTargets = t .* tr.trainMask{1};
valTargets = t .* tr.valMask{1};
testTargets = t .* tr.testMask{1};
trainPerformance = perform(net,trainTargets,y)
valPerformance = perform(net,valTargets,y)
testPerformance = perform(net,testTargets,y)


%%%%% Ejemplo 
Masa_min=64.1394e-9; Masa_max=30*64.1394e-9;
Masas =[0 Masa_min Masa_max]';
 
n_points=30000; 
f_ini=4.965e6;%frecuencia inicial
f_final=4.975e6;%frecuencia final
Lm=64.1394e-3;
Cm=16.0371e-15;
Rm=11.42;
C0=43.3903e-12;
Lmass1=Masas(1); Lmass2=Masas(2);Lmass3=Masas(3);
    coupling=linspace(0.001,0.005,5);
    Lk=coupling(2)*Lm;
    parameters=[Lm,Cm,Rm,C0,Lk,Lmass1,Lmass2,Lmass3];
    fsim=linspace(f_ini,f_final,n_points);
    [Zsim] = New_simulate_Y_4resonators_singleLcoupling_model(parameters,fsim);
    Ysim=1./Zsim; 
    Delta_f=fsim(2)-fsim(1);
     g=abs(Ysim)';
   [p , x, ancho ]=findpeaks(g, 'MinPeakProminence', 1e-3);
   Num_picos=length(x);
   t=-4:5; ti=linspace(-4,5,10000);
   for i=1:Num_picos  
        y=interp1(t, g(x(i)+t), ti, 'spline');
        Fi=fsim(x(i))+ti*(Delta_f);
        [Amp_max, ind_max]=max(y);
        F_max=Fi(ind_max);
        Entrada(1,i*3-2:i*3)=[Amp_max, F_max, ancho(i)];
   end
 
Entrada_Red  = (Entrada -Media_Input)./Std_Input
Salida = net(Entrada_Red').*Std_Output' + Media_Output'
fprintf("Error absoluto: %e\n", Masas - Salida)
% View the Network
view(net)

if (false)
    % Generate MATLAB function for neural network for application
    % deployment in MATLAB scripts or with MATLAB Compiler and Builder
    % tools, or simply to examine the calculations your trained neural
    % network performs.
    genFunction(net,'myNeuralNetworkFunction');
    y = myNeuralNetworkFunction(x);
end
if (false)
    % Genera MATLAB function
    genFunction(net,'myNeuralNetworkFunction','MatrixOnly','yes');
    y = myNeuralNetworkFunction(x);
end
if (false)
    % Genera Simulink 
    gensim(net);
end
