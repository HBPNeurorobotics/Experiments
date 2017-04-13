#!/bin/bash

# This script builds one RPM packages for the template experiments.
#
# Author: Stefan Deser


# First fetch the names from the respective .spec files
CONFIG_PKG_NAME=$(grep Name hbp-configs.spec | awk '{print $2}')


BUILD_DIR=build-config-pkg
RPM_TARGET_DIR=target

# The following directories are the ones which are used in the spec file for
# the models package.
# Here we copy the models over to TMP_DIR_CONF (for use in Gazebo) as well as
# This directory is used in the spec file of the configs package.
# Note that we currently store the models as well as the config folders in the
# same directory. This is subject to change in later versions.
TMP_DIR_CONF=tmp/configs

# The RPM package repositories
KERNEL_RELEASE=`uname -r`
RED_HAT_VERSION=""
if [[ "$KERNEL_RELEASE" =~ ^.+\.el7\..*$ ]]; then
  RED_HAT_VERSION="-el7"
fi
# If RED_HAT_VERSION is empty, we use the RHEL-6 repositories
TESTING_REPOSITORY="neurorobotics-testing$RED_HAT_VERSION"
RELEASE_REPOSITORY="neurorobotics$RED_HAT_VERSION"

# Filter the arguments for the script. Also see:
# http://stackoverflow.com/questions/192249/how-do-i-parse-command-line-arguments-in-bash
#
# We do initialize the values with "0" and only in case the respective argument is
# passed on the commandline we set it to "1". This can the later be used in order
# to react accordingly (e.g. upload the packages to a remote repository).
RELEASE=0

while [[ $# > 0 ]]
do
key="$1"

case $key in
    --release)
    RELEASE=1
    ;;
    *)
    # unknown option, ignore
    ;;
esac
shift # past argument or value
done


function log() {
    echo "[HBP-NRP] $(date --rfc-3339=seconds) $*"
}

function clean_up() {
    log "Cleaning up (old) build files"
    rm -rf ./$BUILD_DIR
    rm -rf ./$TMP_DIR_CONF
    rm -rf ./$RPM_TARGET_DIR
}

# The copying is done in order to keep the .spec file's file section clean
# and simple. We have hardcoded values here which may change, but the .spec
# file should not be affected!
# Note furthermore that the large blender folders are removed.
function copy_configuration_folders() {
    log "Establish a clean directory"
    rm -rf ./$TMP_DIR_CONF/*
    mkdir -p ./$TMP_DIR_CONF
    log "Copy config folders"
    FOLDERS=$(ls -d ../* | egrep -v "_rpmbuild|hbp-scxml|*.xsd")

    for folder in ${FOLDERS[@]}; do
        log "Copying $folder"
        cp -r $folder/ ./$TMP_DIR_CONF
    done
}

# Builds the package specified as argument and cleans up the build directory
# afterwards.
function build_rpm_package() {
    local package_name=$1
    log "Building $package_name package now ..."

    log "Checking if .spec file exists"
    if [ ! -f $package_name.spec ]; then
      log "The .spec file for $package_name does not exist! Exiting now!"
      exit -1
    fi

    # -v  verbose
    # -bb build binary
    # The --define sets the build directory
    rpmbuild -v -bb $package_name.spec --define "_topdir ${PWD}/$BUILD_DIR"
    # Copy built package to target directory
    mkdir -p ./$RPM_TARGET_DIR
    cp ./$BUILD_DIR/RPMS/noarch/$package_name*.rpm ./$RPM_TARGET_DIR
    rm -rf ./$BUILD_DIR
}

function check_if_package_exists() {
    local package_name=$1
    log "Checking if package $package_name does exist"
    ls ./$RPM_TARGET_DIR/$package_name*.noarch.rpm 1> /dev/null 2>&1
    if ! [ $? -eq 0 ]; then
      log "Build unsuccessful: The package $package_name does not exist!"
      # Return a non-zero value indicating that the rpm does not exist
      exit -1
    fi
    # In case everything went fine
    log "Build of package $package_name successfully finished!"
    return 0
}

# Only upload the packages to any repository in case we have a release build,
# i.e. a build that is not a gerrit build, but has been triggered by a gerrit
# submit.
# To which repository the package will be uploaded in the end is decided by
# looking at the version of the built rpm.
function upload_packages_to_repository() {

  # not a release, don't upload any packages
  if [[ $RELEASE -ne 1 ]] ; then
    return 0
  fi

  # determine the type of release based on the RPMs (already verified to exist)
  local is_dev=0
  local is_staging=0
  for rpm in `find ./$RPM_TARGET_DIR -name "*.rpm"` ; do
    # check the release tag .el6 for staging or <number>.el6 for dev
    [[ "`rpm -qp --queryformat '%{RELEASE}' $rpm`" == \.* ]] && is_staging=1 || is_dev=1
  done

  # make sure we don't have a mix of dev and staging RPMs
  if [[ $is_dev -eq 1 && $is_staging -eq 1 ]] ; then
    echo "Build unsuccessful: Mismatched dev/staging RPM versions!"
    exit -1
  elif [[ $is_dev -eq 0 && $is_staging -eq 0 ]] ; then
    echo "Build unsuccessful: Could not determine dev or staging release from RPM versions!"
    exit -1
  fi

  # pick the correct upload repository
  upload_repository=$TESTING_REPOSITORY
  [[ $is_staging -eq 1 ]] && upload_repository=$RELEASE_REPOSITORY

  # Upload the packages which have just been built
  # For a single package one can also use the command:
  #   upload2pulp -r neurorobotics -f my_package.rpm
  #
  # This command is provided by infra, also see:
  # https://bbpteam.epfl.ch/project/issues/servicedesk/customer/portal/3/HELP-3705
  log "Uploading the packages to the repository now ..."
  log "Command: 'upload2pulp -r $upload_repository -d ./$RPM_TARGET_DIR'"
  upload2pulp -r $upload_repository -d ./$RPM_TARGET_DIR
  if ! [ $? -eq 0 ]; then
    log "Build unsuccessful: Uploading packages did not work!"
    # Return a non-zero value indicating that uploading did not work
    exit -1
  fi
}


################################################################################
# Main program
################################################################################
#
# First of all we clean everything up in order to have a sober build.
clean_up
#
# Then we build the two packages.
copy_configuration_folders && build_rpm_package $CONFIG_PKG_NAME
#
# After we have built the package we explicitly check if the rpm packages do
# exist now (since we clean at the beginning, they can only be the new ones).
check_if_package_exists $CONFIG_PKG_NAME \
&& upload_packages_to_repository
