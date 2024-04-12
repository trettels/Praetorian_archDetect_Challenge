import torch
import TrainML
import PraetorianServerConnect as PT
import time

arch_list = ['alphaev56',
             'arm',
             'avr',
             'm68k',
             'mips',
             'mipsel',
             'powerpc',
             's390',
             'sh4',
             'sparc',
             'x86_64',
             'xtensa']
class myNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.layer_stack = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ELU(alpha=0.95),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size),
            torch.nn.ELU(alpha=1.25),
            torch.nn.Softmax(dim=0)
        )
    def forward(self, input):
        logits = self.layer_stack(input)
        return logits

nn = torch.load('B:/ML_data/Praetorian_stats/Network.pt')
with open('B:/ML_data/Praetorian_stats/bytelist.txt','r') as file:
        bytelist = file.readlines()
with open('B:/ML_data/Praetorian_stats/bytecount.txt','r') as file:
      nbytes = int(file.read())
bytelist = [byte.strip('\n') for byte in bytelist]
t0 = time.time()
print('Starting at %s' %(time.strftime("%b.%d.%H:%M")))
sv = PT.Server()
have_hash = False
iter_count = 0
loss_function = torch.nn.BCELoss()
learn_rate = 0.01

while not have_hash:
    iter_count+=1
    sv.get()
    bytestr = str(sv.binary)
    bytestr = bytestr[2:-2]
    test_fragment = TrainML.tensorize_snippet(str(sv.binary), bytelist, nbytes)
    output = nn(test_fragment)
    guess, guess_idx = TrainML.detect_arch(output, arch_list)
    if(guess not in sv.targets):
          print(guess)
          print(sv.targets)
          continue
    sv.post(guess)
    sv.log.info("Guess:[{: >9}]   Answer:[{: >9}]   Wins:[{: >3}]".format(guess, sv.ans, sv.wins))
    if guess != sv.ans:
        optimizer = torch.optim.SGD(nn.parameters(), lr = learn_rate, momentum=0.75)
        optimizer.zero_grad()
        arch_tensor = TrainML.tensorize_arch(arch_list, sv.ans)
        loss = loss_function(output, arch_tensor)
        loss.backward()
        optimizer.step()
    
    print('Iteration: %i :: Elapsed time: %s sec' %(iter_count, time.time() - t0))
    print(str(sv.hash))
    if sv.hash:
        sv.log.info("You win! {}".format(sv.hash))
        have_hash=True
        with open('B:/ML_data/Praetorian_stats/success_hash.txt','w') as file:
            file.write(sv.hash)
