[numFeatures, N] = size(Input_Train);
Media_Input = mean(Input_Train');
Std_Input = std(Input_Train');
Input_Train_Norm=[];

for n=1:numFeatures
    Input_Train_Norm(n,:) = (Input_Train(n,:) - Media_Input(n))./Std_Input(n);
    Input_Test_Norm(n,:) = (Input_Test(n,:) - Media_Input(n))./Std_Input(n);
  %  Input_Val_Norm(n,:) = (Input_Val(n,:) - Media_Input(n))./Std_Input(n);
    Fe_Norm(n,:) = (Fe(n,:) - Media_Input(n))./Std_Input(n);
end

[numOutputs, N] = size(Output_Train);
Media_Output = mean(Output_Train');
Std_Output = std(Output_Train');
Output_Train_Norm=[];
for n=1:numOutputs
    Output_Train_Norm(n,:) = (Output_Train(n,:) - Media_Output(n))./Std_Output(n);
    Output_Test_Norm(n,:) = (Output_Test(n,:) - Media_Output(n))./Std_Output(n);
%    Output_Val_Norm(n,:) = (Output_Val(n,:) - Media_Output(n))./Std_Output(n);
    M_Norm(n,:) = (M(n,:) - Media_Output(n))./Std_Output(n);

end