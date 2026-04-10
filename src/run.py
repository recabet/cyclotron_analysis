from src.scripts import generate_training_data
from src.scripts import train_super_resolution
from src.scripts import test_super_resolution


def main():
    # generate_training_data.main()
    # train_super_resolution.main()
    test_super_resolution.main()


if __name__ == "__main__":
    main()
    
    # nohup python -m src.run 2>&1 | tee train_opt.log &
    
    # FWHM
    # scipy detect peaks compoare psoition
    # increase npoints
    
    # increase the noise ratio
    # fine isotopic composition
    #