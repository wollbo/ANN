%% uppg323
clear all
x=[-5:0.5:5]';
y=[-5:0.5:5]';
nodes = 5;

z = varyNodes(nodes,x,y);

%%

mesh(x,y,z)