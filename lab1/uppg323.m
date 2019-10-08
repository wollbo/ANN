%% uppg323
clear all
close all
x=[-5:0.5:5]';
y=[-5:0.5:5]';
nodes = 24;

%axis([])

z = varyNodes(nodes,x,y);
disp('Finnished')

%%

mesh(x,y,z)