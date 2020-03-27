#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 21:44:04 2018

@author: minoh
"""
#%%
import pandas as pd
import numpy as np
import regex as re
import matplotlib.pyplot as plt
import itertools

#%% import eng.testb corpus and create a token list

tokenList = []
with open('eng.testb', 'r') as f:
    for line in f:
        endword = line.split(' ', 1)[0]
        tokenList.append(endword)
f.close()        
        
#%% with the testb tokens create a plain text to use in NER tools
testText = " ".join(tokenList)

f= open("testb.txt","w+")
f.write(testText)
f.close()
        
#%% list of actual NE tags
actualList = []
with open('eng.testb', 'r') as f:
    for line in f:
        actualList.append(line.split(' ')[-1].strip())
f.close() 

#%% pos tags
pos = []
with open('eng.testb', 'r') as f:
    for line in f:
        if len(line.split(' ')) > 1:
            pos.append((line.split(' ')[1]))
        else:
            pos.append('')
f.close()
#%% chunk tags
chunk =[]
with open('eng.testb', 'r') as f:
    for line in f:
        if len(line.split(' ')) > 1:
            chunk.append((line.split(' ')[2]))
        else:
            chunk.append('')
f.close()
#%% Stanford NER prediction
predictList = []
with open('test_predict_iob', 'r') as f:
    for line in f:
        predictList.append(line.split('\t')[-1].strip())
f.close() 

#%% export file for conlleval
with open("sner.txt", "w") as f:
    f.writelines(map("{} {} {}\n".format, tokenList, actualList, predictList))
  


#%% Illinois NET output
illiOutput = []
with open("illinois-ner/output/testb.txt", 'r') as f:
    for line in f:
        illiOutput = re.split('\s+(?![^\[]*\])', line)

f.close()
#%% convert Illinois NET output to CONLL03 format using IOB style
illiResult = []
tag=''
for i in range(len(illiOutput)):
    item = illiOutput[i]
    if item.startswith(('[PER ', '[LOC ', '[ORG ', '[MISC ')):
        if illiOutput[i-1].startswith(('[PER ', '[LOC ', '[ORG ', '[MISC ')): #if previous item tagged
            if item[:4] == illiOutput[i-1][:4]: #if previous item same tag
                #print(item[:4])
                item = item[1:-1]  
                phrase = item.split(' ')
                tag = phrase[0]
                #print(tag)
                illiResult.append(phrase[1] + ' B-' + tag)
                #print(phrase[1] + ' B-' + tag)
                if len(phrase) > 2:
                    for j in range(2, len(phrase)):
                        illiResult.append(phrase[j] + ' I-' + tag)
            else:
                item = item[1:-1]      
                for word in item.split(' '):
                    if word in ['PER', 'LOC', 'ORG', 'MISC']:
                        tag = 'I-' + word
                    else:
                        illiResult.append(word + ' ' + tag)
        else:
            item = item[1:-1]      
            for word in item.split(' '):
                if word in ['PER', 'LOC', 'ORG', 'MISC']:
                    tag = 'I-' + word
                else:
                    illiResult.append(word + ' ' + tag)
    else:
        illiResult.append(item + ' O')
            
        

#%% create a token list without empty lines
tokWOEmpty = []

for item in tokenList:
    if item != '\n':
        tokWOEmpty.append(item)
        
#%% create a token list from Illinois NET result
illiToken = []
for item in illiResult:
    illiToken.append(item.split(' ')[0])
#%% fix error in token caused by regex split
for i, item in enumerate(illiToken):
    if item.endswith(']') and item != ']':
        illiToken [i] = item[:-1]
        illiToken[i+1] = ']'
        print(i)
        print(illiToken[i])
        print(illiToken[i+1])
        
#%% match tokens between test data and illinois output
popList = []
for i in range(len(tokWOEmpty)):
    if illiToken[i] != tokWOEmpty[i]:
       if illiToken[i] + illiToken[i+1] == tokWOEmpty[i]:
           illiToken[i] = tokWOEmpty[i]
           illiToken.pop(i+1)
           popList.append(i+1)
       elif illiToken[i] + illiToken[i+1] + illiToken[i+2] == tokWOEmpty[i]:
           illiToken[i] = tokWOEmpty[i]
           illiToken.pop(i+1)
           popList.append(i+1)
           illiToken.pop(i+1)
           popList.append(i+1)

illiToken.pop(46666)
popList.append(46666)

for idx in popList:
    illiResult.pop(idx)
#%% check if there is different token
for i in range(len(tokWOEmpty)):
    if illiToken[i] != tokWOEmpty[i]:
        print(i)
        break
    
#%% Illinois NET prediction
illiPredict = []
for item in illiResult:
    illiPredict.append(item.split(' ')[-1].strip())
    
#%% actual list
illiActual = []

for item in actualList:
    if item != '':
        illiActual.append(item)
        
#%% export a file for conlleval
with open("inet.txt", "w") as f:
    f.writelines(map("{} {} {}\n".format, illiToken, illiActual, illiPredict))
  
#%%
illiPos = []

for item in pos:
    if item != '':
        illiPos.append(item) 

illiChunk =[]
for item in chunk:
    if item != '':
        illiChunk.append(item)

#%%
# Compute confusion matrix
cnf_matrix = confusion_matrix(illiActual, CN_predictWOEmpty)
np.set_printoptions(precision=2)
class_names = ['B-LOC','B-MISC', 'B-ORG', 'B-PER', 'I-LOC', 'I-MISC', 'I-ORG', 'I-PER', '0']

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.

    """

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True tag')
    plt.xlabel('Predicted tag')

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')

plt.show()


#%% get unique value
myset = set(illiPredict)
print(myset)

#%% SNER predicted list without empty string
predictWOEmpty = []

for item in predictList:
    if item != '':
        predictWOEmpty.append(item)
        
#%% CN03 predicted list
CN_predict = []
with open('eng.testb_CN03', 'r') as f:
    for line in f:
        CN_predict.append(line.strip())
f.close()  
#%% CN03 predicted list without empty string
CN_predictWOEmpty = []

for item in CN_predict:
    if item != '':
        CN_predictWOEmpty.append(item)
        
#%% 
snerError = []
for i in range(len(illiActual)):
    if illiActual[i] != predictWOEmpty[i]:
        snerError.append((i, illiToken[i], illiActual[i], predictWOEmpty[i], illiPos[i], illiChunk[i]))

#%%
inetError = []
for i in range(len(illiActual)):
    if illiActual[i] != illiPredict[i]:
        inetError.append((i, illiToken[i], illiActual[i], illiPredict[i], illiPos[i], illiChunk[i]))
#%%
CNError = []
for i in range(len(illiActual)):
    if illiActual[i] != CN_predictWOEmpty[i]:
        CNError.append((i, illiToken[i], illiActual[i], CN_predictWOEmpty[i], illiPos[i], illiChunk[i]))

#%% list where all systems had errors. list of tuples (Idx, token, true NER tag, pos tag, chunk tag, SNER, INET, CN03)
sameErrors = []
for i in range(len(snerError)):
    for j in range(len(inetError)):
        for k in range(len(CNError)):
            if snerError[i][0] == inetError[j][0] and inetError[j][0] == CNError[k][0]:
                sameErrors.append((snerError[i][0], snerError[i][1], snerError[i][2],snerError[i][4],snerError[i][5], snerError[i][3], inetError[j][3], CNError[k][3]))
        
#%%
with open("same_error.txt", "w") as f:
    f.write('\n'.join('{}, {}, {}, {}, {}, {}, {}, {}'.format(x[0],x[1], x[2], x[3], x[4], x[5], x[6], x[7]) for x in sameErrors))
#%%
with open("snerError.txt", "w") as f:
    f.write('\n'.join('{}, {}, {}, {}, {}, {}'.format(x[0],x[1], x[2], x[3], x[4], x[5]) for x in snerError))
    
with open("inetError.txt", "w") as f:
    f.write('\n'.join('{}, {}, {}, {}, {}, {}'.format(x[0],x[1], x[2], x[3], x[4], x[5]) for x in inetError))
     
with open("CNError.txt", "w") as f:
    f.write('\n'.join('{}, {}, {}, {}, {}, {}'.format(x[0],x[1], x[2], x[3], x[4], x[5]) for x in CNError))
 
#%% Create error lists
snerOnlyErrors = []
inetOnlyErrors = []
CNOnlyErrors = []
inetErrNum = []
CNErrNum = []
snerErrNum = []

for i in range(len(inetError)):
    inetErrNum.append(inetError[i][0])

for i in range(len(CNError)):
    CNErrNum.append(CNError[i][0])
    
for i in range(len(snerError)):
    snerErrNum.append(snerError[i][0])    

for i in range(len(snerError)):
    if snerError[i][0] not in inetErrNum and snerError[i][0] not in CNErrNum:
        snerOnlyErrors.append(snerError[i])

for i in range(len(inetError)):
    if inetError[i][0] not in snerErrNum and inetError[i][0] not in CNErrNum:
        inetOnlyErrors.append(inetError[i])

for i in range(len(CNError)):
    if CNError[i][0] not in snerErrNum and CNError[i][0] not in inetErrNum:
        CNOnlyErrors.append(CNError[i])

#%%
with open("sner_only_error.txt", "w") as f:
    f.write('\n'.join('{}, {}, {}, {}, {}, {}'.format(x[0],x[1], x[2], x[3], x[4], x[5]) for x in snerOnlyErrors))

with open("inet_only_error.txt", "w") as f:
    f.write('\n'.join('{}, {}, {}, {}, {}, {}'.format(x[0],x[1], x[2], x[3], x[4], x[5]) for x in inetOnlyErrors))
    
with open("CN_only_error.txt", "w") as f:
    f.write('\n'.join('{}, {}, {}, {}, {}, {}'.format(x[0],x[1], x[2], x[3], x[4], x[5]) for x in CNOnlyErrors))

            
#%% 
sner = []
for i in range(len(illiActual)):
    sner.append((i, illiToken[i], illiActual[i], predictWOEmpty[i], illiPos[i], illiChunk[i]))

#%%
inet = []
for i in range(len(illiActual)):
    inet.append((i, illiToken[i], illiActual[i], illiPredict[i], illiPos[i], illiChunk[i]))
#%%
CN = []
for i in range(len(illiActual)):
   CN.append((i, illiToken[i], illiActual[i], CN_predictWOEmpty[i], illiPos[i], illiChunk[i]))

#%% check if a word has inconsistent labels
appearSner = []
for i in range(len(snerOnlyErrors)):
    for j in range(len(sner)):
        if snerOnlyErrors[i][1]  == sner[j][1]:
            appearSner.append(sner[j])
                
for i in range(len(appearSner)-1):
    if appearSner[i][1] == appearSner[i+1][1] and appearSner[i][3] != appearSner[i+1][3]:
        print(appearSner[i])

#%% B-type correctness
bcount = 0
correct = 0
for i in range(len(illiPredict)):
    if illiPredict[i].startswith('B'):
        bcount += 1
        if illiPredict[i] == illiActual[i]:
            correct +=1
print(correct/bcount) #0.39215686274509803

#%%
bcount = 0
correct = 0
for i in range(len(CN_predictWOEmpty)):
    if CN_predictWOEmpty[i].startswith('B'):
        bcount += 1
        if CN_predictWOEmpty[i] == illiActual[i]:
            correct +=1
print(correct/bcount) #0.375


#%%
bcount = 0
correct = 0
for i in range(len(illiPredict)):
    if illiActual[i].startswith('B'):
        bcount += 1
        print(bcount)
        if illiPredict[i] == illiActual[i]:
            correct +=1
            print(correct)
print(correct/bcount) #1.0

#%%
bcount = 0
correct = 0
for i in range(len(CN_predictWOEmpty)):
    if illiActual[i].startswith('B'):
        bcount += 1
        print(bcount)
        if CN_predictWOEmpty[i] == illiActual[i]:
            correct +=1
            print(correct)
print(correct/bcount) #0.3
