import datetime
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.style as mplstyle
import time
import functools

mpl.use('TkAgg')
mplstyle.use('fast')
plt.rcParams["figure.figsize"] = (15,12.5)


RANKS = 2       #max: 4
BANKS = 4       #max: 8
ROWS = 16000    #max: 64000
COLUMNS = 2000  #max: 4000


def vis_single_dimm(path, socket, imc, channel, dimm):
    plt.ion()

    # Load and organize data
    data = pd.read_csv(path, index_col=False, header=0)

    # check data format
    assert set(['socket', 'imc', 'channel', 'dimm', 'row', 'column', 'datetime', 'rank', 'bank']).issubset(set(data.columns))

    data = data.loc[(data['socket'] == socket) & (data['imc'] == imc) & (data['channel'] == channel) & (data['dimm'] == dimm)]

    if len(data) <= 0:
        raise ValueError("Zero datapoints match socket={}, imc={}, channel={} and dimm={}".format(socket, imc, channel, dimm))
    
    data.sort_values(by='datetime', inplace=True)
    start_date = data["datetime"][0]
    
    # Init image data
    img = np.ones((COLUMNS, ROWS)) # image for a single bank
 
    # initialize main view
    sliders = [] # required to protect sliders from garbage collector; every new slider has to be appended to this array
        
    fig, axs = plt.subplots(BANKS, RANKS, sharex=True, sharey=True) # create subplots...
    fig.subplots_adjust(bottom=0.3) # might have to adjust with changing figsize
    fig.suptitle("DIMM #{} \n {} to {}".format(dimm, start_date, start_date), fontsize=15) # ...with title
    aximshows = [[None for _ in range(RANKS)] for _ in range(BANKS)] # array to store imshow returns, required to later update the data they are showing
    buttons = np.array([[None for _ in range(RANKS)] for _ in range(BANKS)])
    # initialize subplots
    for i in range(BANKS):
        for j in range(RANKS):
            aximshows[i][j] = axs[i, j].imshow(img, cmap='gray', vmin=0, vmax=1, origin="lower")
            axs[i, j].title.set_text("Rank:{}, Bank:{}, Errors:0".format(j,i))

            buttons[i][j] = Button(plt.axes([0.25 + (0.08* j % RANKS), 0.25 - (0.025* i % BANKS), 0.075, 0.02]), "Rank:{}; Bank:{}".format(j,i)) # add buttons
            buttons[i][j].label.set_fontsize(9)

    # create sliders and buttons for navigation
    axpos = plt.axes([0.2, 0.0, 0.65, 0.03])
    spos = Slider(axpos, 'Datapoints', 0, len(data["datetime"]), valinit=0,valstep=1)
    sliders.append(spos) # append slider to list outside so it can be referenced inside other functions

    bg = fig.canvas.copy_from_bbox(fig.bbox) # save basic part of the view so it can be easily reloaded on update

    # Function called on button press to create a bank view
    def plt_bank(self, rank, bank):
        print(rank, bank)
        plt.figure((rank+2*bank)+2) # create new figure
        plt.clf() # clear in case the figure already exists; might draw over existing figure otherwise
        ax = plt.axes()


        bank_data = data.loc[(data['rank']==rank) & (data['bank_address']==bank)] # filter data for bank

        fig = plt.imshow(img, cmap='gray', vmin=0, vmax=1, origin="lower")
        
        ax.set_title("Rank:{}, Bank:{}, Errors:0 \n {} to {}".format(rank, bank, start_date, start_date), fontsize=16)

        # define slider for new view
        axpos = plt.axes([0.2, 0.0, 0.65, 0.03])
        spos = Slider(axpos, 'Datapoints', 0, len(bank_data["datetime"]), valinit=0,valstep=1)
        sliders.append(spos)

        # update function for slider
        def update_bank(val):
            start = time.time()

            img = np.zeros((COLUMNS, ROWS))
            img = np.ones((COLUMNS, ROWS))
            tmp = bank_data[:val]
            column = tmp["column"].values
            row = tmp["row"].values
            for i in range(val):
                img[column[i]][row[i]] = 1
                img[column[i]][row[i]] = 0
            fig.set_data(img)

            
            ax.set_title("Rank:{}, Bank:{}, Errors:{} \n {} to {}".format(rank,bank, int(img.size-np.sum(img)), start_date, tmp["datetime"].values[-1]), fontsize=16) # update title
            print(time.time() - start)

        spos.on_changed(update_bank) # connect update function and slider
        

    # update function for slider
    def update(val):
        start = time.time()
        
        fig.canvas.restore_region(bg)
        
        tmp = data[:val] # get data in slider window

        # filter relevant values from data
        column = tmp["column"].values
        row = tmp["row"].values
        rank = tmp["rank"].values
        bank = tmp["bank_address"].values

        # to store the bounds in which errors occur for each bank
        max_bounds = np.ones((BANKS, RANKS, 2))*-1 
        min_bounds = np.ones((BANKS, RANKS, 2))*(ROWS+1)

        # update images based on ddata in slider window
        img = np.zeros((BANKS, RANKS, COLUMNS, ROWS))
        img = np.ones((BANKS, RANKS, COLUMNS, ROWS))
        for i in range(val):
            img[bank[i]][rank[i]][column[i]][row[i]] = 0
            
            # Get the bounds in which the errors are located. Those are 2 points which can be used in the tool to zoom in and view the errors if needed
            min_bounds[bank[i]][rank[i]][1] = min(min_bounds[bank[i]][rank[i]][1], column[i])
            min_bounds[bank[i]][rank[i]][0] = min(min_bounds[bank[i]][rank[i]][0], row[i])
            
            max_bounds[bank[i]][rank[i]][1] = max(max_bounds[bank[i]][rank[i]][1], column[i])
            max_bounds[bank[i]][rank[i]][0] = max(max_bounds[bank[i]][rank[i]][0], row[i])


        # connect updated image data and subplots + update text
        for b in range(BANKS):
            for r in range(RANKS):
                aximshows[b][r].set_data(img[b][r])

                if tuple(max_bounds[b][r])!=(-1,-1) and tuple(min_bounds[b][r])!=((ROWS+1),(ROWS+1)):
                    axs[b][r].title.set_text("Rank:{}, Bank:{}, Errors:{}, Bounds:({},{})({},{})".format(r,b,int(img[b][r].size-np.sum(img[b][r])), int(min_bounds[b][r][0]), int(min_bounds[b][r][1]), int(max_bounds[b][r][0]), int(max_bounds[b][r][1])))
                else: 
                    axs[b][r].title.set_text("Rank:{}, Bank:{}, Errors:{}".format(r,b,int(img[b][r].size-np.sum(img[b][r]))))


        fig.suptitle("DIMM #{} \n {} to {}".format(dimm, start_date, tmp["datetime"].values[-1]), fontsize=15)
        fig.canvas.blit(fig.bbox)
        fig.canvas.draw_idle() # redraw
        print(time.time() - start)

    spos.on_changed(update) # connect update function and slider

    # connect buttons and event functions
    for b in range(BANKS):
        for r in range(RANKS):
            buttons[b][r].on_clicked(functools.partial(plt_bank, rank=r, bank=b))


    plt.show(block=True)
