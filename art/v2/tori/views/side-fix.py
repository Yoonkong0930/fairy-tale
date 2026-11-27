from PIL import Image, ImageDraw; import numpy as np
src='art/v2/tori/views/side-cand1.png'
a=np.array(Image.open(src).convert('RGB')).astype(float)
orig=a.copy()
FUR=np.array([235,189,129.]); SH=np.array([207,146,98.]); EYE=np.array([78,36,19.])
# 1) erase U-line on body side (everything below the collar under-shade, inside body, not cream)
from scipy.ndimage import binary_dilation, gaussian_filter
Y,X=np.mgrid[0:1024,0:1024]
top=(np.sqrt(((orig-FUR)**2).sum(2))>14)&(Y<720)
cov=np.zeros((1024,1024),bool)
for x in range(460,650):
    c=top[600:720,x]; idx=np.where(c)[0]
    if not len(idx): continue
    r=idx[0]
    while r+1<len(c) and c[r+1]: r+=1
    cov[:600+r+1,x]=True
d=np.sqrt(((orig-FUR)**2).sum(2))
body=(np.abs(orig-255).sum(2)>40); cream=(orig.sum(2)>660)
win=np.zeros((1024,1024),bool); win[670:850,468:604]=True
mline=binary_dilation((d>5)&body&~cream&win&~cov,iterations=2)&body&~cream&win&~cov
a[mline]=FUR
print('line px erased',mline.sum())
# 2) eye highlight mirror
ys,xs=np.where(orig[340:480,640:780].sum(2)<250); ex0,ex1,ey0,ey1=xs.min()+640,xs.max()+640,ys.min()+340,ys.max()+340
cx=(ex0+ex1)/2; cy=(ey0+ey1)/2; rx=(ex1-ex0)/2; ry=(ey1-ey0)/2
print('eye',ex0,ex1,ey0,ey1,cx,cy)
Y,X=np.mgrid[0:1024,0:1024]
inner=((X-cx)/(rx-5))**2+((Y-cy)/(ry-5))**2<1
hl=inner&(orig.sum(2)>EYE.sum()+25)
print('hl px',hl.sum())
a[inner]=EYE
hy,hx=np.where(hl)
mx=np.round(2*cx-hx).astype(int)
a[hy,mx]=orig[hy,hx]
# 3) arm: supersampled capsule, shade crescent behind
S=4
def capsule(p0,r0,p1,r1,off=(0,0)):
    im=Image.new('L',(1024*S,1024*S),0); dr=ImageDraw.Draw(im)
    p0=np.array(p0)+off; p1=np.array(p1)+off
    n=60
    for t in np.linspace(0,1,n):
        c=p0*(1-t)+p1*t; r=r0*(1-t)+r1*t
        dr.ellipse([(c[0]-r)*S,(c[1]-r)*S,(c[0]+r)*S,(c[1]+r)*S],fill=255)
    return np.array(im.resize((1024,1024),Image.LANCZOS)).astype(float)/255
P0=(546,684);P1=(600,806)
shadow=capsule(P0,33,P1,30,off=(-3,4))
armtop=capsule(P0,33,P1,30,off=(5,-5))
# clip shoulder under collar: keep arm only below collar bottom: find collar rows already drawn -> just clip y<672
top=(np.sqrt(((orig-FUR)**2).sum(2))>14)&(Y<720)   # collar + its under-shade stay on top
cov=np.zeros((1024,1024),bool)
for x in range(460,650):
    c=top[600:720,x]; idx=np.where(c)[0]
    if not len(idx): continue
    r=idx[0]
    while r+1<len(c) and c[r+1]: r+=1
    cov[:600+r+1,x]=True
from scipy.ndimage import gaussian_filter
k=1-np.clip(gaussian_filter(cov.astype(float),0.7),0,1)
shadow*=k; armtop*=k
# arm shoulder blends into body: fade shading near shoulder top so no hard edge
a=a*(1-shadow[...,None])+SH*shadow[...,None]
a=a*(1-armtop[...,None])+FUR*armtop[...,None]
Image.fromarray(np.clip(a,0,255).astype('uint8')).save('art/v2/tori/views/side-original.png')
diff=np.abs(a-orig).sum(2)>6; ys,xs=np.where(diff); print('changed bbox',xs.min(),xs.max(),ys.min(),ys.max(),diff.sum())
