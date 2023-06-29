#!/usr/bin/python
#
#  Copyright 2002-2022 Barcelona Supercomputing Center (www.bsc.es)
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import datetime

from rocrate.rocrate import ROCrate
from rocrate.model.person import Person
from rocrate.model.contextentity import ContextEntity
from rocrate.model.entity import Entity
from rocrate.model.file import File
from rocrate.utils import iso_now

from pathlib import Path
from urllib.parse import urlsplit

import yaml
import os
import uuid
import typing
import datetime as DT

CRATE = ROCrate()


def add_file_not_in_crate(in_url: str) -> None:
    """
    When adding local files that we don't want to be physically in the Crate, they must be added with a file:// URI
    CAUTION: If the file has been already added (e.g. for INOUT files) add_file won't succeed in adding a second entity
    with the same name

    :param in_url: File added as input or output, but not in the RO-Crate

    :returns: None
    """

    # method_time = time.time()
    url_parts = urlsplit(in_url)
    final_item_name = os.path.basename(in_url)
    file_properties = {
        "name": final_item_name,
        "sdDatePublished": iso_now(),
        "dateModified": DT.datetime.utcfromtimestamp(os.path.getmtime(url_parts.path)).replace(microsecond=0).isoformat(),  # Schema.org
    }  # Register when the Data Entity was last accessible

    if url_parts.scheme == "file":  # Dealing with a local file
        file_properties["contentSize"] = os.path.getsize(url_parts.path)
        # add_file_time = time.time()
        CRATE.add_file(
            in_url,
            fetch_remote=False,
            validate_url=False,  # True fails at MN4 when file URI points to a node hostname (only localhost works)
            properties=file_properties,
        )
        # add_file_time = time.time() - add_file_time

    elif url_parts.scheme == "dir":  # DIRECTORY parameter
        # For directories, describe all files inside the directory
        has_part_list = []
        for root, dirs, files in os.walk(
            url_parts.path, topdown=True
        ):  # Ignore references to sub-directories (they are not a specific in or out of the workflow),
            # but not their files
            dirs.sort()
            files.sort()
            for f_name in files:
                listed_file = os.path.join(root, f_name)
                dir_f_url = "file://" + url_parts.netloc + listed_file
                has_part_list.append({"@id": dir_f_url})
                dir_f_properties = {
                    "name": f_name,
                    "sdDatePublished": iso_now(),  # Register when the Data Entity was last accessible
                    "dateModified": DT.datetime.utcfromtimestamp(os.path.getmtime(url_parts.path)).replace(microsecond=0).isoformat(),
                    # Schema.org
                    "contentSize": os.path.getsize(listed_file),
                }
                CRATE.add_file(
                    dir_f_url,
                    fetch_remote=False,
                    validate_url=False,
                    # True fails at MN4 when file URI points to a node hostname (only localhost works)
                    properties=dir_f_properties,
                )
        file_properties["hasPart"] = has_part_list
        CRATE.add_dataset(
            fix_dir_url(in_url), properties=file_properties
        )  # fetch_remote and validate_url false by default. add_dataset also ensures the URL ends with '/'

    else:  # Remote file, currently not supported in COMPSs. validate_url already adds contentSize and encodingFormat
        # from the remote file
        CRATE.add_file(in_url, validate_url=True, properties=file_properties)

    # print(f"Method vs add_file TIME: {time.time() - method_time} vs {add_file_time}")


def get_main_entities(wf_info: dict) -> typing.Tuple[str, str, str]:
    """
    Get COMPSs version and mainEntity from dataprovenance.log first lines
    3 First lines expected format: compss_version_number\n main_entity\n output_profile_file\n
    Next lines are for "accessed files" and "direction"
    mainEntity can be directly obtained for Python, or defined by the user in the YAML (sources_main_file)

    :param wf_info: YAML dict to extract info form the application, as specified by the user

    :returns: COMPSs version, main COMPSs file name, COMPSs profile file name
    """

    # Build the whole source files list in list_of_sources, and get a backup main entity, in case we can't find one
    # automatically. The mainEntity must be an existing file, otherwise the RO-Crate won't have a ComputationalWorkflow
    list_of_sources = []
    sources_list = []
    # Should contain absolute paths, for correct comparison (two files in different directories
    # could be named the same)
    main_entity = None
    backup_main_entity = None
    if "files" in wf_info:
        files_list = []
        if isinstance(wf_info["files"], list):
            files_list = wf_info["files"]
        else:
            files_list.append(wf_info["files"])
        for file in files_list:
            path_file = Path(file).expanduser()
            resolved_file = str(path_file.resolve())
            if path_file.exists():
                list_of_sources.append(resolved_file)
        # list_of_sources = wf_info["files"].copy()
        if list_of_sources:  # List of files not empty
            backup_main_entity = list_of_sources[0]
            # Assign first file name as mainEntity
    if "sources_dir" in wf_info:
        if isinstance(wf_info["sources_dir"], list):
            sources_list = wf_info["sources_dir"]
        else:
            sources_list.append(wf_info["sources_dir"])
        for source in sources_list:
            path_sources = Path(source).expanduser()
            if not path_sources.exists():
                print(f"PROVENANCE | WARNING: Specified path in ro-crate-info.yaml 'sources_dir' does not exist ({path_sources})")
                continue
            resolved_sources = str(path_sources.resolve())
            # print(f"resolved_sources is: {resolved_sources}")
            for root, dirs, files in os.walk(resolved_sources, topdown=True):
                for f_name in files:
                    # print(f"PROVENANCE DEBUG | ADDING FILE to list_of_sources: {f_name}. root is: {root}")
                    full_name = os.path.join(root, f_name)
                    list_of_sources.append(full_name)
                    if backup_main_entity is None and Path(f_name).suffix in {
                        ".py",
                        ".java",
                        ".jar",
                        ".class",
                    }:
                        backup_main_entity = full_name
                        # print(
                        #     f"PROVENANCE DEBUG | FOUND SOURCE FILE AS BACKUP MAIN: {backup_main_entity}"
                        # )

    print(f"PROVENANCE | Number of source files detected: {len(list_of_sources)}")
    # print(f"PROVENANCE DEBUG | Source files detected: {list_of_sources}")

    # Can't get backup_main_entity from sources_main_file, because we do not know if it really exists
    if backup_main_entity is None:
        print(
            f"PROVENANCE | ERROR: Unable to find application source files. Please, review your "
            f"ro_crate_info.yaml definition ('sources_dir', and 'files' terms)"
        )
        raise FileNotFoundError
    # print(f"PROVENANCE DEBUG | backup_main_entity is: {backup_main_entity}")

    with open(dp_log, "r") as f:
        compss_v = next(f).rstrip()  # First line, COMPSs version number
        second_line = next(f).rstrip()
        # Second, main_entity. Use better rstrip, just in case there is no '\n'
        if second_line.endswith(".py"):
            # Python. Line contains only the file name, need to locate it
            detected_app = second_line
        else:  # Java app. Need to fix filename first
            # Translate identified main entity matmul.files.Matmul to a comparable path
            me_sub_path = second_line.replace(".", "/")
            detected_app = me_sub_path + ".java"
        # print(f"PROVENANCE DEBUG | Detected app is: {detected_app}")

        for file in list_of_sources:  # Try to find the identified mainEntity
            if file.endswith(detected_app):
                # print(
                #     f"PROVENANCE DEBUG | IDENTIFIED MAIN ENTITY FOUND IN LIST OF FILES: {file}"
                # )
                main_entity = file
                break
        # main_entity has a value if mainEntity has been automatically detected
        if "sources_dir" in wf_info and "sources_main_file" in wf_info:
            # Check what the user has defined
            # if sources_main_file is an absolute path, the join has no effect
            found = False
            for source in sources_list:  # Created before
                path_sources = Path(source).expanduser()
                if not path_sources.exists():
                    continue
                resolved_sources = str(path_sources.resolve())
                resolved_sources_main_file = os.path.join(resolved_sources, wf_info["sources_main_file"])
                if any(file == resolved_sources_main_file for file in list_of_sources):
                    # The file exists
                    # print(
                    #     f"PROVENANCE DEBUG | The file defined at sources_main_file exists: "
                    #     f" {resolved_sources_main_file}"
                    # )
                    if resolved_sources_main_file != main_entity:
                        print(
                            f"PROVENANCE | WARNING: The file defined at sources_main_file "
                            f"({resolved_sources_main_file}) in ro-crate-info.yaml does not match with the "
                            f"automatically identified mainEntity ({main_entity})"
                        )
                    # else: the user has defined exactly the file we found
                    # In both cases: set file defined by user
                    main_entity = resolved_sources_main_file
                    # Can't use Path, file may not be in cwd
                    found = True
                    break
            if not found:
                print(
                    f"PROVENANCE | WARNING: the defined 'sources_main_file' ({wf_info['sources_main_file']}) does "
                    f"not exist in the defined 'sources_dir' ({wf_info['sources_dir']}). Check your ro-crate-info.yaml."
                )
                # If we identified the mainEntity automatically, we select it when the one defined
                # by the user is not found

        if main_entity is None:
            # When neither identified, nor defined by user: get backup
            main_entity = backup_main_entity
            print(
                f"PROVENANCE | WARNING: the detected mainEntity {detected_app} does not exist in the list "
                f"of application files provided in ro-crate-info.yaml. Setting {main_entity} as mainEntity"
            )

        third_line = next(f).rstrip()
        out_profile_fn = Path(third_line)

    return compss_v, main_entity, out_profile_fn.name


def process_accessed_files() -> typing.Tuple[list, list]:
    """
    Process all the files the COMPSs workflow has accessed. They will be the overall inputs needed and outputs
    generated of the whole workflow.
    - If a task that is an INPUT, was previously an OUTPUT, it means it is an intermediate file, therefore we discard it
    - Works fine with COLLECTION_FILE_IN, COLLECTION_FILE_OUT and COLLECTION_FILE_INOUT

    :returns: List of Inputs and Outputs of the COMPSs workflow
    """

    inputs = set()
    outputs = set()

    with open(dp_log, "r") as f:
        for line in f:
            file_record = line.rstrip().split(" ")
            if len(file_record) == 2:
                if (
                    file_record[1] == "IN" or file_record[1] == "IN_DELETE"
                ):  # Can we have an IN_DELETE that was not previously an OUTPUT?
                    if (
                        file_record[0] not in outputs
                    ):  # A true INPUT, not an intermediate file
                        inputs.add(file_record[0])
                    #  Else, it is an intermediate file, not a true INPUT or OUTPUT. Not adding it as an input may
                    # be enough in most cases, since removing it as an output may be a bit radical
                    #     outputs.remove(file_record[0])
                elif file_record[1] == "OUT":
                    outputs.add(file_record[0])
                else:  # INOUT, COMMUTATIVE, CONCURRENT
                    if (
                        file_record[0] not in outputs
                    ):  # Not previously generated by another task (even a task using that same file), a true INPUT
                        inputs.add(file_record[0])
                    # else, we can't know for sure if it is an intermediate file, previous call using the INOUT may
                    # have inserted it at outputs, thus don't remove it from outputs
                    outputs.add(file_record[0])
            # else dismiss the line

    l_ins = list(inputs)
    l_ins.sort()
    l_outs = list(outputs)
    l_outs.sort()

    print(f"PROVENANCE | INPUTS({len(l_ins)})")
    print(f"PROVENANCE | OUTPUTS({len(l_outs)})")

    return l_ins, l_outs


def fix_dir_url(in_url: str) -> str:
    """
    Fix dir:// URL returned by the runtime, change it to file:// and ensure it ends with '/'

    :param in_url: URL that may need to be fixed

    :returns: A file:// URL
    """

    runtime_url = urlsplit(in_url)
    if (
        runtime_url.scheme == "dir"
    ):  # Fix dir:// to file:// and ensure it ends with a slash
        new_url = "file://" + runtime_url.netloc + runtime_url.path
        if new_url[-1] != "/":
            new_url += "/"  # Add end slash if needed
        return new_url
    else:
        return in_url  # No changes required


def add_file_to_crate(
    file_name: str,
    compss_ver: str,
    main_entity: str,
    out_profile: str,
    ins: list,
    outs: list,
    in_sources_dir: str,
) -> None:
    """
    Get details of a file, and add it physically to the Crate. The file will be an application source file, so,
    the destination directory should be 'application_sources/'

    :param file_name: File to be added physically to the Crate, full path resolved
    :param compss_ver: COMPSs version number
    :param main_entity: COMPSs file with the main code, full path resolved
    :param out_profile: COMPSs application profile output
    :param ins: List of input files
    :param outs: List of output files
    :param in_sources_dir: Path to the defined sources_dir. May be passed empty

    :returns: None
    """

    file_path = Path(file_name)
    file_properties = dict()
    file_properties["name"] = file_path.name
    file_properties["contentSize"] = os.path.getsize(file_name)
    # Check file extension, to decide how to add it in the Crate file_path.suffix
    # if file_path.suffix == ".jar":  # We can ignore main_entity
    #     namespace = main_entity_in.rstrip().split(".")
    #     print(f"namespace: {namespace}")
    #     main_entity = namespace[0] + ".jar"  # Rebuild package name
    # else:  # main_file.py or any other file
    #     main_entity = main_entity_in
    # print(f"main_entity is: {main_entity}, file_path is: {file_path}")

    # main_entity has its absolute path, as well as file_name
    if file_name == main_entity:
        file_properties["description"] = "Main file of the COMPSs workflow source files"
        if file_path.suffix == ".jar":
            file_properties["encodingFormat"] = (
                [
                    "application/java-archive",
                    {"@id": "https://www.nationalarchives.gov.uk/PRONOM/x-fmt/412"},
                ],
            )
            # Add JAR as ContextEntity
            CRATE.add(
                ContextEntity(
                    CRATE,
                    "https://www.nationalarchives.gov.uk/PRONOM/x-fmt/412",
                    {"@type": "WebSite", "name": "Java Archive Format"},
                )
            )
        elif file_path.suffix == ".class":
            file_properties["encodingFormat"] = (
                [
                    "application/java",
                    {"@id": "https://www.nationalarchives.gov.uk/PRONOM/x-fmt/415"},
                ],
            )
            # Add CLASS as ContextEntity
            CRATE.add(
                ContextEntity(
                    CRATE,
                    "https://www.nationalarchives.gov.uk/PRONOM/x-fmt/415",
                    {"@type": "WebSite", "name": "Java Compiled Object Code"},
                )
            )
        else:  # .py, .java, .c, .cc, .cpp
            file_properties["encodingFormat"] = "text/plain"
        if complete_graph.exists():
            file_properties["image"] = {
                "@id": "complete_graph.svg"
            }  # Name as generated
        file_properties["input"] = []
        for item in ins:
            file_properties["input"].append({"@id": fix_dir_url(item)})
        file_properties["output"] = []
        for item in outs:
            file_properties["output"].append({"@id": fix_dir_url(item)})

    else:
        # Any other extra file needed
        file_properties["description"] = "Auxiliary File"
        if file_path.suffix == ".py" or file_path.suffix == ".java":
            file_properties["encodingFormat"] = "text/plain"
        elif file_path.suffix == ".json":
            file_properties["encodingFormat"] = [
                "application/json",
                {"@id": "https://www.nationalarchives.gov.uk/PRONOM/fmt/817"},
            ]
        elif file_path.suffix == ".pdf":
            file_properties["encodingFormat"] = (
                [
                    "application/pdf",
                    {"@id": "https://www.nationalarchives.gov.uk/PRONOM/fmt/276"},
                ],
            )
        elif file_path.suffix == ".svg":
            file_properties["encodingFormat"] = (
                [
                    "image/svg+xml",
                    {"@id": "https://www.nationalarchives.gov.uk/PRONOM/fmt/92"},
                ],
            )
        elif file_path.suffix == ".jar":
            file_properties["encodingFormat"] = (
                [
                    "application/java-archive",
                    {"@id": "https://www.nationalarchives.gov.uk/PRONOM/x-fmt/412"},
                ],
            )
            # Add JAR as ContextEntity
            CRATE.add(
                ContextEntity(
                    CRATE,
                    "https://www.nationalarchives.gov.uk/PRONOM/x-fmt/412",
                    {"@type": "WebSite", "name": "Java Archive Format"},
                )
            )
        elif file_path.suffix == ".class":
            file_properties["encodingFormat"] = (
                [
                    "Java .class",
                    {"@id": "https://www.nationalarchives.gov.uk/PRONOM/x-fmt/415"},
                ],
            )
            # Add CLASS as ContextEntity
            CRATE.add(
                ContextEntity(
                    CRATE,
                    "https://www.nationalarchives.gov.uk/PRONOM/x-fmt/415",
                    {"@type": "WebSite", "name": "Java Compiled Object Code"},
                )
            )

    # Build correct dest_path. If the file belongs to sources_dir, need to remove all "sources_dir" from file_name,
    # respecting the sub_dir structure.
    # If the file is defined individually, put in the root of application_sources

    if in_sources_dir:
        # /home/bsc/src/file.py must be translated to application_sources/src/file.py,
        # but in_sources_dir is /home/bsc/src
        # print(f"in_sources_dir is {in_sources_dir}")
        # list_root = list(Path(in_sources_dir).parts)
        # list_root.pop()
        # new_root = os.path.join(*list_root)
        new_root = str(Path(in_sources_dir).parents[0])
        # print(f"new_root is {new_root}")
        final_name = file_name[len(new_root) + 1 :]
        # print(f"final_name is {final_name}")
        path_in_crate = "application_sources/" + final_name
    else:
        path_in_crate = "application_sources/" + file_path.name

    # print(f"path_in_crate: {path_in_crate}")

    if file_name != main_entity:
        # print(f"PROVENANCE DEBUG | Adding auxiliary source file: {file_name}")
        CRATE.add_file(
            source=file_name, dest_path=path_in_crate, properties=file_properties
        )
    else:
        # We get lang_version from dataprovenance.log
        # print(            f"PROVENANCE DEBUG | Adding main source file: {file_path.name}, file_name: {file_name}")
        CRATE.add_workflow(
            source=file_name,
            dest_path=path_in_crate,
            main=True,
            lang="COMPSs",
            lang_version=compss_ver,
            properties=file_properties,
            gen_cwl=False,
        )

        # complete_graph.svg
        if complete_graph.exists():
            file_properties = dict()
            file_properties["name"] = "complete_graph.svg"
            file_properties["contentSize"] = complete_graph.stat().st_size
            file_properties["@type"] = ["File", "ImageObject", "WorkflowSketch"]
            file_properties[
                "description"
            ] = "The graph diagram of the workflow, automatically generated by COMPSs runtime"
            # file_properties["encodingFormat"] = (
            #     [
            #         "application/pdf",
            #         {"@id": "https://www.nationalarchives.gov.uk/PRONOM/fmt/276"},
            #     ],
            # )
            file_properties["encodingFormat"] = (
                    [
                        "image/svg+xml",
                        {"@id": "https://www.nationalarchives.gov.uk/PRONOM/fmt/92"},
                    ],
            )
            file_properties["about"] = {
                "@id": path_in_crate
            }  # Must be main_entity_location, not main_entity alone
            # Add PDF as ContextEntity
            # CRATE.add(
            #     ContextEntity(
            #         CRATE,
            #         "https://www.nationalarchives.gov.uk/PRONOM/fmt/276",
            #         {
            #             "@type": "WebSite",
            #             "name": "Acrobat PDF 1.7 - Portable Document Format",
            #         },
            #     )
            # )
            CRATE.add(
                ContextEntity(
                    CRATE,
                    "https://www.nationalarchives.gov.uk/PRONOM/fmt/92",
                    {
                        "@type": "WebSite",
                        "name": "Scalable Vector Graphics",
                    },
                )
            )
            CRATE.add_file(complete_graph, properties=file_properties)
        else:
            print(
                f"PROVENANCE | WARNING: complete_graph.svg file not found. "
                f"Provenance will be generated without image property"
            )

        # out_profile
        if os.path.exists(out_profile):
            file_properties = dict()
            file_properties["name"] = out_profile
            file_properties["contentSize"] = os.path.getsize(out_profile)
            file_properties["description"] = "COMPSs application Tasks profile"
            file_properties["encodingFormat"] = [
                "application/json",
                {"@id": "https://www.nationalarchives.gov.uk/PRONOM/fmt/817"},
            ]
            # Add JSON as ContextEntity
            CRATE.add(
                ContextEntity(
                    CRATE,
                    "https://www.nationalarchives.gov.uk/PRONOM/fmt/817",
                    {"@type": "WebSite", "name": "JSON Data Interchange Format"},
                )
            )
            CRATE.add_file(out_profile, properties=file_properties)
        else:
            print(
                f"PROVENANCE | WARNING: COMPSs application profile has not been generated. \
                  Make sure you use runcompss with --output_profile=file_name"
                f"Provenance will be generated without profiling information"
            )

        # compss_command_line_arguments.txt
        file_properties = dict()
        file_properties["name"] = "compss_command_line_arguments.txt"
        file_properties["contentSize"] = os.path.getsize(
            "compss_command_line_arguments.txt"
        )
        file_properties[
            "description"
        ] = "COMPSs command line execution command, including parameters passed"
        file_properties["encodingFormat"] = "text/plain"
        CRATE.add_file("compss_command_line_arguments.txt", properties=file_properties)


def main():
    # First, read values defined by user from ro-crate-info.yaml
    try:
        with open(info_yaml, "r", encoding="utf-8") as fp:
            try:
                yaml_content = yaml.safe_load(fp)
            except yaml.YAMLError as exc:
                print(exc)
                raise exc
    except IOError:
        with open("ro-crate-info_TEMPLATE.yaml", "w", encoding="utf-8") as ft:
            template = """COMPSs Workflow Information:
  name: Name of your COMPSs application
  description: Detailed description of your COMPSs application
  license: Apache-2.0  # Provide better a URL, but these strings are accepted:
            # https://about.workflowhub.eu/Workflow-RO-Crate/#supported-licenses
  sources_dir: [path_to/dir_1, path_to/dir_2]  # Optional: List of directories containing application source files. 
            # Relative or absolute paths can be used
  sources_main_file: my_main_file.py  # Optional  Name of the main file of the application, located in one of the 
            # sources_dir. Relative paths from a sources_dir or absolute paths can be used
  files: [main_file.py, aux_file_1.py, aux_file_2.py] # List of application files
            # Relative or absolute paths can be used
Authors:
  - name: Author_1 Name
    e-mail: author_1@email.com
    orcid: https://orcid.org/XXXX-XXXX-XXXX-XXXX
    organisation_name: Institution_1 name
    ror: https://ror.org/XXXXXXXXX # Find them in ror.org
  - name: Author_2 Name
    e-mail: author2@email.com
    orcid: https://orcid.org/YYYY-YYYY-YYYY-YYYY
    organisation_name: Institution_2 name
    ror: https://ror.org/YYYYYYYYY # Find them in ror.org
            """
            ft.write(template)
            print(
                f"PROVENANCE | ERROR: YAML file ro-crate-info.yaml not found in your working directory. A template"
                f" has been generated in file ro-crate-info_TEMPLATE.yaml"
            )
        raise

    # Get Sections
    compss_wf_info = yaml_content["COMPSs Workflow Information"]
    authors_info_yaml = yaml_content["Authors"]  # Now a list of authors
    authors_info = []
    if isinstance(authors_info_yaml, list):
        authors_info = authors_info_yaml
    else:
        authors_info.append(authors_info_yaml)

    # COMPSs Workflow RO Crate generation

    # Root Entity
    CRATE.name = compss_wf_info["name"]
    CRATE.description = compss_wf_info["description"]
    CRATE.license = compss_wf_info["license"]  # Faltarà el detall de la llicència????
    authors_set = set()
    organisations_set = set()
    for author in authors_info:
        authors_set.add(author["orcid"])
        organisations_set.add(author["ror"])
        CRATE.add(
            Person(
                CRATE,
                author["orcid"],
                {
                    "name": author["name"],
                    "contactPoint": {"@id": "mailto:" + author["e-mail"]},
                    "affiliation": {"@id": author["ror"]},
                },
            )
        )
        CRATE.add(
            ContextEntity(
                CRATE,
                "mailto:" + author["e-mail"],
                {
                    "@type": "ContactPoint",
                    "contactType": "Author",
                    "email": author["e-mail"],
                    "identifier": author["e-mail"],
                    "url": author["orcid"],
                },
            )
        )
        CRATE.add(
            ContextEntity(
                CRATE,
                author["ror"],
                {"@type": "Organization", "name": author["organisation_name"]},
            )
        )
    author_list = list()
    for creator in authors_set:
        author_list.append({"@id": creator})
    CRATE.creator = author_list
    org_list = list()
    for org in organisations_set:
        org_list.append({"@id": org})
    CRATE.publisher = org_list

    # print(f"compss_wf_info at the beginning: {compss_wf_info}")

    # Get mainEntity from COMPSs runtime report dataprovenance.log
    compss_ver, main_entity, out_profile = get_main_entities(compss_wf_info)
    print(
        f"PROVENANCE | COMPSs version: {compss_ver}, main_entity is: {main_entity}, out_profile is: {out_profile}"
    )

    # Process set of accessed files, as reported by COMPSs runtime.
    # This must be done before adding the Workflow to the RO-Crate

    part_time = time.time()
    ins, outs = process_accessed_files()
    print(
        f"PROVENANCE | RO-CRATE data_provenance.log processing TIME (process_accessed_files): "
        f"{time.time() - part_time} s"
    )

    # Add files that will be physically in the crate
    part_time = time.time()
    # print(f"compss_wf_info: {compss_wf_info}")
    added_files = []
    if "sources_dir" in compss_wf_info:
        # Optional, the user specifies a directory with all sources
        sources_list = []
        if isinstance(compss_wf_info["sources_dir"], list):
            sources_list = compss_wf_info["sources_dir"]
        else:
            sources_list.append(compss_wf_info["sources_dir"])
        for source in sources_list:
            path_sources = Path(source).expanduser()
            if not path_sources.exists():
                continue
            resolved_sources = str(path_sources.resolve())
            # print(f"resolved_sources is: {resolved_sources}")
            # resolved_sources = compss_wf_info["sources_dir"]
            for root, dirs, files in os.walk(resolved_sources, topdown=True):
                for f_name in files:
                    # print(f"Adding file from sources_dir: root: {root} f_name: {f_name}")
                    resolved_file = os.path.join(root, f_name)
                    add_file_to_crate(
                        resolved_file,
                        compss_ver,
                        main_entity,
                        out_profile,
                        ins,
                        outs,
                        resolved_sources,
                    )
                    added_files.append(resolved_file)

    if "files" in compss_wf_info:
        # print(f"compss_wf_info(files): {compss_wf_info['files']}")
        files_list = []
        if isinstance(compss_wf_info["files"], list):
            files_list = compss_wf_info["files"]
        else:
            files_list.append(compss_wf_info["files"])
        for file in files_list:
            path_file = Path(file).expanduser()
            resolved_file = str(path_file.resolve())
            if not path_file.exists():
                print(
                    f"PROVENANCE | WARNING: A file defined as 'files' in ro-crate-info.yaml does not exist "
                    f"({resolved_file})"
                )
                continue
            if resolved_file not in added_files:
                # print(f"Adding file from 'files': {file}")
                add_file_to_crate(
                    resolved_file, compss_ver, main_entity, out_profile, ins, outs, ""
                )
                added_files.append(resolved_file)
            else:
                print(
                    f"PROVENANCE | WARNING: A file addition was attempted twice in 'files' and 'sources_dir': "
                    f"{resolved_file}"
                )
    print(
        f"PROVENANCE | RO-CRATE adding physical files TIME (add_file_to_crate): {time.time() - part_time} s"
    )

    # Add files not to be physically in the Crate
    part_time = time.time()
    for item in ins:
        add_file_not_in_crate(item)
    print(
        f"PROVENANCE | RO-CRATE adding input files' references TIME (add_file_not_in_crate): "
        f"{time.time() - part_time} s"
    )

    part_time = time.time()
    for item in outs:
        add_file_not_in_crate(item)
    print(
        f"PROVENANCE | RO-CRATE adding output files' references TIME (add_file_not_in_crate): "
        f"{time.time() - part_time} s"
    )

    # Dump to file
    part_time = time.time()
    folder = "COMPSs_RO-Crate_" + str(uuid.uuid4()) + "/"
    CRATE.write(folder)
    print(f"PROVENANCE | COMPSs RO-Crate created successfully in subfolder {folder}")
    print(f"PROVENANCE | RO-CRATE dump TIME: {time.time() - part_time} s")
    # cleanup from workingdir
    os.remove("compss_command_line_arguments.txt")


if __name__ == "__main__":
    import sys
    import time

    exec_time = time.time()

    # Usage: python /path_to/generate_COMPSs_RO-Crate.py ro-crate-info.yaml /path_to/dataprovenance.log
    if len(sys.argv) != 3:
        print(
            "PROVENANCE | Usage: python /path_to/generate_COMPSs_RO-Crate.py "
            "ro-crate-info.yaml /path_to/dataprovenance.log"
        )
        exit()
    else:
        info_yaml = sys.argv[1]
        dp_log = sys.argv[2]
        path_dplog = Path(sys.argv[2])
        complete_graph = path_dplog.parent / "monitor/complete_graph.svg"
    main()

    print(
        f"PROVENANCE | RO-CRATE GENERATION TOTAL EXECUTION TIME: {time.time() - exec_time} s"
    )
