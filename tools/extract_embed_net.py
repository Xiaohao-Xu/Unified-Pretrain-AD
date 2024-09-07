import torch
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", metavar="DIR", help="run directory")
    args, opts = parser.parse_known_args()
    for i in range(102): ## TO REVISE, JUST HACK
        try:
            pthfile = args.run_dir + "/epoch_"+str(i)+".pth"
            save_pth=pthfile.replace("epoch", "latest_trans").replace(".pth","_e.pth")
            net = torch.load(pthfile,map_location=torch.device('cpu'))
            for k in  list(net["state_dict"].keys()):
                if "fuser" in k:
                    del net["state_dict"][k]
            del net["optimizer"]
            del net["meta"]
            torch.save(net,save_pth)
        except:
            pass

if __name__ == "__main__":
    main()
              
