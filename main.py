import json


keyplen=18
credit=0.5

def fpath(kind,a):
    kind=str(kind)
    num=str(a)
    if len(num)<12:
        d=12-len(num)
        num='0'*d+num
    prestr="result"+kind+"\\json\\"+kind+"_"+num+"_keypoints.json"
    return prestr


import numpy as np

def mdis(a,b):
    vis=[0]*len(b);
    sumc=0
    sumdis=0

    for i in range(keyplen):
        mpos=-1
        mv=1e9
        x=a[i][0]
        y=a[i][1]
        
        for j in range(keyplen):
            if vis[j]==1:
                continue
            if np.sqrt((x-b[j][0])**2+(y-b[j][1])**2)<mv:
                mv=np.sqrt((x-b[j][0])**2+(y-b[j][1])**2)
                mpos=j
        assert mpos!=-1
        sumdis+=mv*a[i][2]*b[mpos][2]
        
        vis[mpos]=1
        sumc+=a[i][2]*b[mpos][2]
    if sumc<0.6:
        return 1e9;
    return sumdis/sumc
def dis(a,b,debug=0):
    #a,b is a pos of keypoint of a person(keyplen,3)
    #return mdis(a,b)
    times=5
    ans=1e9
    import random
    for z in range(times):
        ans=min(ans,mdis(a,b))
        random.shuffle(a)
    
    return ans

def center(a):

    x=0
    y=0
    num=0
    for j in range(keyplen):
        if a[j][2]<credit:
            continue
        x+=a[j][0]
        y+=a[j][1]
        num+=1
    if num==0:
        return 1e9,1e9
    x/=num
    y/=num
    
    return x,y


def inran(x,l,r):
    return x>=l and x<=r





def count_dis(kind):
    res_path="res_of_dis"+str(kind)+".txt"
    final_res=[]

    #here is range for across
    xl=790
    xr=1050
    yl=190
    yr=1000
    with open(res_path,'w') as ff:
        ff.seek(0)
        ff.truncate()
        max_len=[-1,7496,7058][kind]
        print("maxlen is "+str(max_len))
        lim=600
        pre_credit=0
        credit_rate=[]
        for i in range(0,max_len+1):#for each flame
            with open(fpath(kind,i),'r') as f:
                temp = json.loads(f.read())
                group=[]

                tmp_credit=0
                credit_ave=1
                num=0
                for item in temp["people"]:
                    ans=np.zeros((keyplen,3),dtype=np.float)
                    count=0
                    for j in range(len(item["pose_keypoints_2d"])//3):
                        x=item["pose_keypoints_2d"][j*3]
                        y=item["pose_keypoints_2d"][j*3+1]
                        c=item["pose_keypoints_2d"][j*3+2]
                        ans[j][0]=x
                        ans[j][1]=y
                        ans[j][2]=c
                        #as long as the preson's point in range, consider count
                        if inran(x,xl,xr) and inran(y,yl,yr):
                            tmp_credit+=c
                            credit_ave+=1

                        if c>credit:
                            count+=1
                    if count<4:
                        continue
                    group.append(ans)
                    num+=1
                debug=0
                #tmp_credit/=credit_ave#jisuan ave for credit
                credit_rate.append([i,tmp_credit])
                pre_credit=tmp_credit

                '''
                count for dis of person
                '''
                for k in range(num):#count for the change rate for credit

                    for p in range(k+1,num):
                        import copy
                        t1=copy.deepcopy(group[k])
                        t2=copy.deepcopy(group[p])
                        res=dis(group[k],group[p],debug)
                        group[k]=t1
                        group[p]=t2
                        
                        cx1,cy1=center(group[k])
                        cx2,cy2=center(group[p])
                        pos1=str(cx1)+","+str(cy1)+" "
                        pos2=str(cx2)+","+str(cy2)+" "
                        dis2=np.sqrt((cx1-cx2)**2+(cy1-cy2)**2)
                        if debug:
                            print(pos1+pos2)
                        if res<lim and inran(cx1,xl,xr) and inran(cx2,xl,xr) and inran(cy1,yl,yr) and inran(cy2,yl,yr) and abs(cx1-cx2)<200:
                            #meaning this two person is folloing
                            final_res.append(i)
                            ff.write("on flame "+str(i)+" has with dis "+str(res)+" with "+pos1+pos2+"dis2 is "+str(dis2)+"\n")
                            ff.write("while this time,num of people is "+str(num)+"\n")

    return final_res,credit_rate


ress,res_for_credit=count_dis(2)#if 1,taopiao ,2 is normal

import matplotlib.pyplot as plt
import numpy as np
a=np.array([x[0] for x in res_for_credit])#x alex
b=np.array([x[1] for x in res_for_credit])




k=3
timelim=100


ans=[]
i=0
j=0
def count_variance(a):
    #a is array
    num=len(a)
    if num==0:
        return 0
    tot=0
    for i in a:
        tot+=i
    ave=tot/num
    res=0
    for i in a:
        res+=(i-ave)**2
    return res/num

while j<len(ress) and i<len(ress):
    while j+1<len(ress) and ress[j+1]-ress[i]<timelim:
        j+=1
    if j-i+1>=k and count_variance([x[0] for x in res_for_credit[ress[i]:ress[j]+1]])>2.5:
        ans.append([ress[i],ress[j]])
        j+=1
        i=j
    else:
        i+=1
        j=i

print(ans)


plt.plot(a,b)

ax2 = plt.twinx()
y=[]
for time in a:
    sud=0
    for rg in ans:
        if inran(time,rg[0],rg[1]):
            sud=1
            break
    y.append(sud)

ax2.plot(a,np.array(y),'r')

plt.show()


