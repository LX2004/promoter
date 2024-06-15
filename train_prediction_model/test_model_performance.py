from random import choice
import numpy as np
from utils import *
from net import predict_transformerv2
from IPython import display

def compute_subsititution_absolute(a):
    
    means = []
    for arr in a:

        abs_arr = np.abs(arr)
        mean_arr = np.mean(abs_arr)
        means.append(mean_arr)

    return np.array(means)

def compute_scaler(model_output):

    model_output = model_output.detach().cpu().numpy()

    max_reads =  6.440191125726022
    min_reads =  0.9380190974762103

    model_output = min_reads + (max_reads - min_reads) * model_output
    return model_output

def encode_list(seqList):
    
    X_seq=np.array([Dimer_split_seqs(''.join(np.char.join('', sequence))) for sequence in seqList])
    return X_seq

def mutationEffects(randomset,model):

    randomset_=encode_list(randomset)
    print(randomset_.shape)

    randomset_ = torch.tensor(randomset_).to(device)
    df_pred=compute_scaler(model(randomset_)) #Computes predictions for the random set

    repdic=dict([(b,np.repeat(b,N)) for b in bases])
    rec=dict([(b1,dict([(b2,[]) for b2 in bases if b2!=b1])) for b1 in bases])

    for i in range(seqlen):
        for j,b1 in enumerate(bases):

            b1_idx=np.where(randomset[:,i]==repdic[b1])

            randomset_mut=randomset[b1_idx] 

            for k,b2 in enumerate(bases):
                if b2!=b1: #mutate to all different bases
                    randomset_mut[:,i]=repdic[b2][b1_idx]
                    randomset_mut_=encode_list(randomset_mut)

                    randomset_mut_ = torch.tensor(randomset_mut_).to(device)

                    Ei=compute_scaler(model(randomset_mut_))-df_pred[b1_idx]

                    rec[b1][b2].append(np.squeeze(Ei)) #record effect of mutations
    
    return rec

def plt_Ei(rec,save_path=""):

    values = []
    names = []

    fig, axes=plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True, figsize=(12,8))

    for i, axlist in enumerate(axes):

        for j, ax in enumerate(axlist):

            if i==0:
                ax.set_xlabel("%s" % (bases[j]), fontsize=12)
                ax.xaxis.set_label_position('top') 

            if j==0:
                ax.set_ylabel("%s" % (bases[i]), fontsize=12)

            if i!=j:

                if (bases[i] in ["A","T"] and bases[j] in ["A","T"]) or (bases[i] in ["C","G"] and bases[j] in ["C","G"]):
                    #highlight transitions vs transversion
                    ax.set_facecolor((229/255,229/255,229/255))#RGB tuple

                else:
                    ax.set_facecolor((229/255,229/255,229/255))#RGB tuple

                values.append(compute_subsititution_absolute(a=rec[bases[i]][bases[j]]))
                names.append(f'{bases[i]} to {bases[j]}')

                box_parts = ax.boxplot(rec[bases[i]][bases[j]],  0, '')

                for pc in box_parts["whiskers"]:
                    pc.set_linestyle("solid")
                    pc.set_linewidth(0.5)

                for pc in box_parts["boxes"]:
                    pc.set_linewidth(0.5)

                for pc in box_parts["caps"]:
                    pc.set_linewidth(0.5)

                for pc in box_parts["means"]:
                    pc.set_linewidth(0.5)



                ax.set_xlim(0, seqlen+1)
                ax.set_xticks(range(1, seqlen+1, 10))

                ax.set_xticklabels(map(str, range(-seqlen, 0, 10)))
                
                ax.set_ylim((-0.4,0.4))
                ax.set_yticks(np.arange(-0.5,0.5,0.2))

    fig.subplots_adjust(hspace=0,wspace=0)

    if save_path:
        fig.savefig(save_path)

    plt.show()

    return values, names

if __name__=='__main__':

    bases=["A","T","G","C"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.cuda.set_device(1)
    print('device =',device)

    N=200
    seqlen=50

    nat_promoter = np.load('../data/promoter.npy')
    promoters = nat_promoter.tolist()

    randomset = []

    for promoter in  promoters[0:N]:

        lst = [char.upper() for char in promoter]
        randomset.append(lst)

    randomset = np.array(randomset)

    model = torch.load('model/kfold_predict_415_mertric=0.29523156.pth').to(device)

    rec = mutationEffects(randomset,model)
    values, names = plt_Ei(rec,save_path='figures/000.svg')

    color1 = (78/255,98/255,171/255)
    color2 = (70/255,158/255,180/255)
    color3 = (135/255,207/255,164/255)

    linewidth = 5
    markersize = 16

    fig, axs = plt.subplots(1, 4, figsize=(50, 10))

    k = 0
    for ax in axs:

        for i in range(k,k+3):

            label=names[i]

            if label[0] in ['A','T'] and label[-1] in ['G','C']:

                ax.plot([x for x in range(-50,0,1)],values[i], label=label,linewidth=linewidth,  marker='^', markersize=markersize)

            elif label[-1] in ['A','T'] and label[0] in ['G','C']:

                ax.plot([x for x in range(-50,0,1)],values[i], label=label ,linewidth=linewidth,  marker='s', markersize=markersize)
            else:
                ax.plot([x for x in range(-50,0,1)],values[i], label=label, linestyle='--' ,linewidth=linewidth,  marker='o', markersize=markersize)
        
        ax.legend(fontsize=40,loc='upper right')
        ax.tick_params(labelsize=40)
        
        ax.set_xlabel('Distance to TSS',fontsize=40)
        ax.set_ylabel('Changes in promoter strength',fontsize=40)

        ax.set_xlim([-14, -4])
        ax.set_ylim([0, 0.18])

        k += 3

    plt.tight_layout()
    plt.savefig('figures/subsititution_by_absolute111.svg')
    plt.show()


    fig, axs = plt.subplots(1, 4, figsize=(48, 10))

    k = 0
    for ax in axs:

        for i in range(k,k+3):

            label=names[i]

            if label[0] in ['A','T'] and label[-1] in ['G','C']:

                ax.plot([x for x in range(-50,0,1)],values[i], label=label,linewidth=linewidth,  marker='^', markersize=markersize)

            elif label[-1] in ['A','T'] and label[0] in ['G','C']:

                ax.plot([x for x in range(-50,0,1)],values[i], label=label ,linewidth=linewidth,  marker='s', markersize=markersize)
            else:
                ax.plot([x for x in range(-50,0,1)],values[i], label=label, linestyle='--' ,linewidth=linewidth,  marker='o', markersize=markersize)
        
        ax.legend(fontsize=40,loc='upper left')
        ax.tick_params(labelsize=40)
        
        ax.set_xlabel('Distance to TSS',fontsize=40)
        ax.set_ylabel('Changes in promoter strength',fontsize=40)

        ax.set_xlim([-40, -30])
        ax.set_ylim([0, 0.12])

        k += 3

    plt.tight_layout()
    plt.savefig('figures/subsititution_by_absolute222.svg')
    plt.show()