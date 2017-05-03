Summary: Configuration files for the HBP NRP (ExD, BIBI)
Name: hbp-configs
Version: 1.2.1
Release: 0%{?dist}
BuildArch: noarch
License: GPL

%description
This package contains the Experiment Designer config files as well as the Brain
and Body Integrator (BIBI) config files and h5 brain models for the HBP
Neurorobotics Platform.

%prep

%build

%install
rm -rf %{buildroot}
mkdir -p %{buildroot}/opt/hbp/gazebo
cp -r ../../tmp/configs %{buildroot}/opt/hbp/gazebo/models
chmod -R o+rx %{buildroot}/opt/hbp
# Note that the configuration data has nothing to do with the gazebo models,
# but for historical reasons the configuration data and the models have to
# be located in the same directory. This is subject to change in newer versions
# of this package.
#
# The first two lines above have to be replaced with the following lines then:
#
# mkdir -p %{buildroot}/opt/hbp
# cp -r ../../tmp/configs %{buildroot}/opt/hbp
#
# and the build script has to change the tmp directory for this package.

%post
chmod -R o+rx /opt/hbp

%files
/opt/hbp
