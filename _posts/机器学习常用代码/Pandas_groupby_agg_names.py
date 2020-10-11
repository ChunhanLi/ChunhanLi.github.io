te = df.groupby('A').agg({'B':{'Mean': np.mean, 'Sum': np.sum},'C':{'Mean':np.mean}})
col = []
for a,b in te.columns:
    col.append(a+'_'+b)
te.columns = col

###
tmp = df.groupby(f1, as_index=False)[f2].agg({'{}_{}_list'.format(f1, f2): list})