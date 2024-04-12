import torch
from pathlib import Path
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

## To Do / ideas:
##    - Get byte counts across archs, get anova / manova / mutual info for byte vs. arch
##    - make word-removal list to remove least informative (most common, and singletons[?])
##    - add dummy-word for "removed word"
##    - Set up plot to map gradient / cell weight shifts
##    - Other network designs

class myNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.layer_stack = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
           # torch.nn.ELU(alpha=0.95),
           # torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size),
            torch.nn.ELU(alpha=1.25),
            torch.nn.Softmax(dim=0)
        )
    def forward(self, input):
        logits = self.layer_stack(input)
        return logits


            

def collect_snippets(base_dir, arch_dir):
    arch_path = base_dir + arch_dir + '/'
    snippet_list = list()
    filelist = Path(arch_path).glob('*')
    for ff in filelist:
        bytestr = ff.read_text()
        bytestr = bytestr[2:-2]
        snippet_list.append(bytestr)
    return snippet_list

def split_train_validate_test(arch_dict, frac_train, frac_validate):
    train_set = dict()
    validate_set = dict()
    test_set = dict()
    for arch in arch_dict.keys():
        snippet_list = arch_dict[arch]
        train_ct = int(frac_train * len(snippet_list))
        validate_ct = int(frac_validate * len(snippet_list))
        test_ct = len(snippet_list) - train_ct - validate_ct
        train_set[arch] = list()
        test_set[arch] = list()
        validate_set[arch] = list()
        for ii in range(train_ct):
            train_set[arch].append(snippet_list.pop(random.randint(0,len(snippet_list)-1)))
        for ii in range(validate_ct):
            validate_set[arch].append(snippet_list.pop(random.randint(0,len(snippet_list)-1)))
        for ii in range(test_ct):
            test_set[arch].append(snippet_list.pop(random.randint(0,len(snippet_list)-1)))
    return train_set, validate_set, test_set

def build_unique_wordlist(arch_dict, max_words, use_frac):
    uniq_wordlist = list()
    all_snippets = arch_dict.values()
    for arch in arch_dict.keys():
        all_snippets = arch_dict[arch]
        for snippet in all_snippets:
            split_bytestr = snippet.split('\\x')
            if len(split_bytestr) > max_words:
                max_words = len(split_bytestr)
            for byte in split_bytestr:
                if not byte.strip() and not byte.strip() in uniq_wordlist:
                    uniq_wordlist.append(byte.strip())
                elif not byte.strip():
                    continue
                uniq_wordlist.append(byte.strip())
    wordlist = Counter(uniq_wordlist)
    trimmed_wordlist = list()
    for key, val in wordlist.items():
        if val > use_frac * len(uniq_wordlist):
            trimmed_wordlist.append(key)
    uniq_wordlist = sorted(list(set(trimmed_wordlist)))
    return uniq_wordlist, max_words

def tensorize_snippet(snippet, uniq_wordlist, max_words):
    uniq_wordlist = sorted(uniq_wordlist)
    snippet_words = list()
    split_bytestr = snippet.split('\\x')
    for b64_byte in split_bytestr:
            snippet_words.append(b64_byte.strip())
    snippet_tensor = torch.zeros(max_words, device="cuda")
    snippet_words = snippet_words[1:]    
    for idx, word in enumerate(snippet_words):
        if not word:
            continue
        try:
            uniq_wordlist.index(word.strip())
        except ValueError:
            continue
        word_idx = uniq_wordlist.index(word)
        snippet_tensor[idx] = word_idx / len(uniq_wordlist)
    return snippet_tensor

def tensorize_arch(arch_list, arch):
    arch_tensor = torch.zeros(len(arch_list), device="cuda")
    arch_tensor[arch_list.index(arch)] = 1
    return arch_tensor    

def rand_snippet(arch_list, snip_dict, uniq_wordlist, max_words):
    arch_idx = random.randint(0,len(arch_list)-1)
    arch = arch_list[arch_idx]
    arch_tensor = tensorize_arch(arch_list, arch)
    snip_list = snip_dict[arch]
    snip_idx = random.randint(0, len(snip_list)-1)
    snip = snip_list[snip_idx]
    snip_tensor = tensorize_snippet(snip, uniq_wordlist, max_words)
    return snip, arch, snip_tensor, arch_tensor


def train_net(mynn, arch_tensor, snippet_tensor, criterion, learn_rate):
    optimizer = torch.optim.SGD(mynn.parameters(), lr = learn_rate, momentum=0.75)
    optimizer.zero_grad()
    output = mynn(snippet_tensor)
    loss = criterion(output, arch_tensor)
    if loss > 10.0:
        print('Excessive loss increase, reducing learning rate')
        learn_rate = learn_rate * 0.5
    loss.backward()
    optimizer.step()
    return output, loss.item(), learn_rate

def detect_arch(output, arch_list):
    ton_n, top_i = output.topk(1)
    archidx = top_i[0].item()
    return arch_list[archidx], archidx

def main():
    device=('cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu')
    
    training_data_basedir = 'B:/ML_Data/Praetorian/'
    num_hidden = 1024
    learn_rate = 0.02
    loss_function = torch.nn.BCELoss()
    plot_iter_modulo = 5000
    validate_iter_modulo = 2500
    numiter = 400000
    num_validate_iter = 2000
    num_test_iter = 2000
    test_iter_modulo = 10000
    max_words=42
    use_frac = 0.01

    arch_dict = dict()
    # scan folder, get archs
    arch_set = sorted(Path(training_data_basedir).glob('*'))

    # for each arch, collect words & process
    for arch in arch_set:
        arch_str = str(arch).split('\\')
        arch_str = arch_str[-1]
        arch_dict[str(arch_str)] = collect_snippets(training_data_basedir, str(arch_str))
    wordlist, max_words = build_unique_wordlist(arch_dict, max_words, use_frac)
    print('Byte list length: %i' %(len(wordlist)))
    arch_list = sorted(list(arch_dict.keys()))
    # split datasets into train/validate/test, ~60-30-10 ratio
    train, validate, test = split_train_validate_test(arch_dict,0.4, 0.3)
    
    # initialize neural network: RNN, LSTM, or Clockwork-RNN?
    #network = MyRNN(len(wordlist), num_hidden, len(list(arch_dict.keys()))).to("cuda")
    network = myNN(max_words, num_hidden, len(arch_list)).to(device)
    # Train network, plotting every plot_iter_modulo iterations
    current_loss = 0
    all_loss=list()
    print('Beginning training!')
    t0 = time.time()
    
    fig, ax = plt.subplots(figsize=(9,6))
    plt.show(block=False)
    x=np.linspace(0,numiter, int(1 + numiter/plot_iter_modulo))
    y=np.zeros(x.shape)
    x2=np.linspace(0,numiter, int(1 + numiter/validate_iter_modulo))
    y2=np.zeros(x2.shape)
    (ln,) = ax.plot(x,y, '-ro', animated=True)
    (ln2,) = ax.plot(x2,y2, '-bx', animated=True)
    plt.ylim(-0.01, 0.5)
    plt.xlim(0, numiter)
    plt.pause(0.1)
    bg=fig.canvas.copy_from_bbox(fig.bbox)
    ax.draw_artist(ln)
    fig.canvas.blit(fig.bbox)
    yidx=0
    y2idx=0
    success_count=0
    for iter in range(1, numiter+1):
        snip, arch, snip_tensor, arch_tensor = rand_snippet(arch_list, train, wordlist, max_words)
        output, loss, learn_rate = train_net(network, arch_tensor, snip_tensor, loss_function, learn_rate)
        current_loss += loss
        guess, guess_idx = detect_arch(output, arch_list)
        if guess == arch:
            success_count +=1
        if iter % plot_iter_modulo == 0:
            result_str = '\u001b[1;32;40m<<Correct>>\u001b[0;0m' if guess == arch else '\u001b[1;31;40m<<Incorrect>> (%s)\u001b[0;0m' % arch
            print('%s :: Iteration: %d Completion: %d%% :: e.time: %d seconds :: Learning Rate: %.2f\n\t\tResults>> Mean Success Rate: %.2f%% -- Loss: %.4f -- %s %s' %(time.strftime("%b.%d.%H:%M"), iter, iter/numiter*100, time.time() - t0, learn_rate, 100*success_count/plot_iter_modulo, loss, guess, result_str))
            
            with open('B:/ML_data/Praetorian_stats/log.txt','a') as file:
                file.writelines('%s :: Iteration: %d Completion: %d%% :: e.time: %d seconds :: Learning Rate: %.2f ::  Results>> Mean Success Rate: %.2f%% -- Loss: %.4f -- %s %s' %(time.strftime("%b.%d.%H:%M"), iter, iter/numiter*100, time.time() - t0, learn_rate, 100*success_count/plot_iter_modulo, loss, guess, result_str))
            
            all_loss.append(current_loss / plot_iter_modulo)
            y[yidx] = current_loss / plot_iter_modulo
            current_loss = 0
            success_count = 0
            yidx +=1
            fig.canvas.restore_region(bg)
            ln.set_ydata(y)
            ax.draw_artist(ln)
            ax.draw_artist(ln2)
            fig.canvas.blit(fig.bbox)
            fig.canvas.flush_events()
            fname = 'B:/ML_data/Praetorian_stats/Network_inProgress.pt'
            torch.save(network,fname)

        if iter % validate_iter_modulo == 0:
            validate_loss_mean = 0
            v_success_mean=0
            v_learning_rate = learn_rate * 2
            for iter2 in range(num_validate_iter):
                snip, arch, snip_tensor, arch_tensor = rand_snippet(arch_list, validate, wordlist, max_words)
                output, loss, v_learning_rate = train_net(network, arch_tensor, snip_tensor, loss_function, v_learning_rate)
                guess, guess_idx = detect_arch(output, arch_list)
                validate_loss_mean += loss / num_validate_iter
                if guess == arch:
                    v_success_mean += 1 / num_validate_iter
            validate_result_str = '\u001b[1;33;40m<<VALIDATION SET>>\u001b[0;0m Learning Rate: %.2f :: Mean Success Rate: %.2f%%, Mean Loss: %.4f' %( learn_rate, 100 * v_success_mean, validate_loss_mean)
            print(validate_result_str)
            with open('B:/ML_data/Praetorian_stats/log.txt','a') as file:
                file.write(validate_result_str + '\n')
            y2[y2idx] = validate_loss_mean
            current_loss = 0
            success_count = 0
            y2idx +=1
            fig.canvas.restore_region(bg)
            ln2.set_ydata(y2)
            ax.draw_artist(ln)
            ax.draw_artist(ln2)
            fig.canvas.blit(fig.bbox)
            fig.canvas.flush_events()
            
        if iter % test_iter_modulo == 0:
            test_loss_mean = 0
            t_success_mean=0
            t_learning_rate = learn_rate * 1.5
            for iter2 in range(num_test_iter):
                snip, arch, snip_tensor, arch_tensor = rand_snippet(arch_list, test, wordlist, max_words)
                output, loss, t_learning_rate = train_net(network, arch_tensor, snip_tensor, loss_function, t_learning_rate)
                guess, guess_idx = detect_arch(output, arch_list)
                test_loss_mean += loss / num_test_iter
                if guess == arch:
                    t_success_mean += 1 / num_test_iter
            test_result_str = '\u001b[1;35;40m<<TEST SET>>\u001b[0;0m Learning Rate: %.2f :: Mean Success Rate: %.2f%%, Mean Loss: %.4f' %( learn_rate, 100 * t_success_mean, test_loss_mean)
            print(test_result_str)
            with open('B:/ML_data/Praetorian_stats/log.txt','a') as file:
                file.write(test_result_str + '\n')

    # Save network weights
    fig.savefig('B:/ML_data/Praetorian_stats/LossPlot.png')
    torch.save(network,'B:/ML_data/Praetorian_stats/Network.pt')
    ## Save unique_wordlist for use later
    with open('B:/ML_data/Praetorian_stats/bytelist.txt','w') as file:
        for word in wordlist:
            file.write(word +'\n')
    with open('B:/ML_data/Praetorian_stats/bytecount.txt','w') as file:
        file.write(str(max_words))


if __name__ == "__main__":
    main()
