from src.scripts import generate_cluster_const_noise
from src.scripts import train_cluster_prof_v2
from src.scripts import test_cluster_super_resolution


def main():

        # Step 1: Generate Clustered Dataset with Constant Noise
        generate_cluster_const_noise.main()
    
        # Step 2: Train Super-Resolution Model on Clustered Dataset
        train_cluster_prof_v2.main()
    
        # Step 3: Test Super-Resolution Model on Clustered Dataset
        test_cluster_super_resolution.main()


if __name__ == "__main__":
    main()
    
    # nohup python -m src.run 2>&1 | tee train_opt.log &

    
    