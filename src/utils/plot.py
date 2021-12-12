import pandas as pd

def plot_after(ts, component_id=None, start_time=None, label=""):
    if component_id != None: ts = ts.univariate_component(component_id)
    if start_time:
        start_time = pd.Timestamp(start_time)
        if start_time < ts.start_time(): start_time = ts.start_time()
    else:
        start_time = ts.start_time()
    ts_after = ts.drop_before(start_time)
    ts_after.plot(label=label)