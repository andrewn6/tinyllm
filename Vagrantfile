Vagrant.configure("2") do |config|
  # Define the master container
  config.vm.define "master" do |master|
    master.vm.provider "docker" do |docker, override|
      docker.build_dir = "."          # Path to the directory with Dockerfile
      docker.dockerfile = "Dockerfile.vagrant"  # Specify custom Dockerfile
      override.vm.box = nil           # No base box needed for Docker provider
      docker.has_ssh = true           # Ensure SSH access
      docker.privileged = true        # Optional: grant elevated privileges

      master.vm.hostname = 'master'
    end
  end

  # Define the slave container
  config.vm.define "slave" do |slave|
    slave.vm.provider "docker" do |docker, override|
      docker.build_dir = "."
      docker.dockerfile = "Dockerfile.vagrant"  # Specify custom Dockerfile
      override.vm.box = nil
      docker.has_ssh = true
      docker.privileged = true

      slave.vm.hostname = 'slave'
    end
  end
end

