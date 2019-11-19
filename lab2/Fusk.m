%fusk
load votes.dat;
load mpparty.dat;
load mpsex.dat;
load mpdistrict.dat

votes_re = reshape(votes,[349 31]);

net = selforgmap([10 10],'topologyFcn','gridtop')
net = train(net,votes)
view(net)
y = net(votes);
classes = vec2ind(y);
