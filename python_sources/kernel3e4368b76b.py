#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <title>Page Title</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.0/css/bootstrap.min.css">
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.0/js/bootstrap.min.js"></script>
    <style>
            .fontSizeOfth {
                font-size: 14px;
                padding-top: 0%;
            }
        
            .table th {
                padding-top: .25rem;
                padding-bottom: .25rem;
            }
        
            .form-group {
                margin-bottom: 0.2rem;
            }
        </style>
</head>
<body>
<div class="container-fluid">
    <div class="row">
        <div class="col-md-6 text-center">
            <span style="color:red;">No Template found. Would you like to create template for this type of form?</span>
            <a href = "{% url 'create_template' %}"><button class="btn-xs btn-success">Create Template</button></a>
            <br>
            <br>
            <iframe id="pdf_file"
                src="{% if details.file %}{% if extension == '.png' or extension == '.jpg' or extension == '.jpeg' %}{% if exists == 0 %}{{ pdf_file_path }}{% else %}{{ pdf_file_path }}{% endif %}{% endif %}{% if extension == '.pdf' %}{{ pdf_file_path }}{% endif %}{% endif %}"
                width="100%" height="{% if num_of_pages == 1 %}100%{% else %}94%{% endif %}">
            </iframe>
        </div>
        <div class="col-md-6" style="overflow-y:scroll;height:100%;">
            <a href="{% url 'dashboard' %}" style="position: absolute;text-decoration: none;"><i
                    class="fa fa-home fa-2x"></i>/Dashboard</a>

            <center>
                <h4 style="padding-top: 1%;">Accounting Form</h4>
            </center>
            <form action="" method="POST">
                {% csrf_token %}
                <input type="text" name="transaction_invoice" value="" hidden>
                <div class="col-12">
                    <div id="other">
                        <div class="row">
                            <div class="col-lg-6">
                                <div class="input-group input-group-sm mb-3">
                                    <div class="input-group-prepend">
                                        <label class="input-group-text" for="invoice_number">Invoice No.<span
                                                style=color:red>*</span></label>
                                    </div>
                                    <input type="text" value="{{ invoice_no }}" class="form-control invoice_no"
                                        name="invoice_no" id="invoice_no" placeholder="Invoice No.">
                                </div>
                            </div>
                            <div class="col-lg-6">
                                <div class="input-group input-group-sm mb-3">
                                    <div class="input-group-prepend">
                                            <label class="input-group-text" for="invoice_date">Invoice Date<span style=color:red>*</span></label>
                                    </div>
                                    <input type="date" id="theDate" class="form-control invoice_date" aria-label="Sizing example input" name="invoice_date" aria-describedby="inputGroup-sizing-default" value="{{ invoice_date }}">
                                </div>
                            </div>
                        </div>
                        <div class="row">

                            <div class="col-lg-6">
                                <div class="input-group input-group-sm mb-3">
                                    <div class="input-group-prepend">
                                        <label class="input-group-text" for="supply_state">Provider/Vendor Name<span style=color:red>*</span></label>
                                    </div>
                                    <input type="text" name="provider_vendor_name" id="provider_vendor_name" value="{{ provider_vendor_name }}"
                                    class="form-control" placeholder="Provider/Vendor Name">
                                </div>
                            </div>
                            <div class="col-lg-6">
                                <div class="input-group input-group-sm mb-3">
                                    <div class="input-group-prepend">
                                        <label class="input-group-text" for="supply_state">Recipient/Ledger Name<span style=color:red>*</span></label>
                                    </div>
                                    <input type="text" name="recipient_ledger_name" id="recipient_ledger_name" value="{{ recipient_ledger_name }}"
                                    class="form-control" placeholder="Recipient/Ledger Name">
                                </div>
                            </div>
                        </div>

                        <div class="row">

                            <div class="col-lg-6">
                                <div class="input-group input-group-sm mb-3">
                                    <div class="input-group-prepend">
                                        <label class="input-group-text" for="supply_state">Place of Supply State<span style=color:red>*</span></label>
                                    </div>
                                    <input type="text" name="place_of_supply_state" id="place_of_supply_state" value="{{ place_of_supply_state }}"
                                    class="form-control" placeholder="Place Of Supply">
                                    <!-- <select class="custom-select supply_state place_of_supply" id="supply_state"
                                        name="state">
                                        {% for element in place_of_supply_state %}
                                        <option value="{{element.state}}"
                                            {% if element.state == 'Haryana' %}selected{% endif %}>
                                            {{element.state}}</option>
                                        {% endfor %}
                                    </select> -->
                                </div>
                            </div>
                            <div class="col-lg-6">
                                <div class="input-group input-group-sm mb-3" id="supply_countryy">
                                    <div class="input-group-prepend">
                                        <label class="input-group-text" for="supply_country">Place of Supply
                                            Country<span style=color:red>*</span></label>
                                    </div>
                                    <select class="custom-select" id="supply_country" name="country">
                                        <option selected>India</option>
                                    </select>
                                </div>
                            </div>

                        </div>

                        <div class="row">
                            <div class="col-lg-6">
                                <div class="input-group input-group-sm mb-3">
                                    <div class="input-group-prepend">
                                        <label class="input-group-text" for="invoice_value">Invoice Value<span
                                                style=color:red>*</span></label>
                                    </div>
                                    <input type="text" name="invoice_value" id="invoice_value" value="{{ invoice_value }}" class="form-control" placeholder="Invoice Value">
                                </div>
                            </div>
                            <div class="col-lg-6">
                                <div class="input-group input-group-sm mb-3">
                                    <div class="input-group-prepend">
                                        <label class="input-group-text" for="taxable_value">Taxable Value<span
                                                style=color:red>*</span></label>
                                    </div>
                                    <input type="text" name="taxable_value" id="taxable_value" value="{{ invoice_value }}"
                                        class="form-control">
                                </div>
                            </div>
                        </div>
                        <div class="row">
                            <div class="col-lg-6">
                                <div class="input-group input-group-sm mb-3">
                                    <div class="input-group-prepend">
                                        <label class="input-group-text" for="gst_no">GST No.
                                        </label>
                                    </div>
                                    <input type="text" name="gst_no" id="gst_no" value="{{ gst_no }}" class="form-control">
                                </div>
                            </div>
                            <div class="col-lg-6">
                                <div class="input-group input-group-sm mb-3">
                                    <div class="input-group-prepend">
                                        <label class="input-group-text" for="invoice_value">IGST
                                        </label>
                                    </div>
                                    <input type="text" name="igst_value" id="igst_value" value="{{ igst_value }}" class="form-control">
                                </div>
                            </div>
                        </div>
                        <div class="row">
                            <div class="col-lg-6">
                                <div class="input-group input-group-sm mb-3">
                                    <div class="input-group-prepend">
                                        <label class="input-group-text" for="invoice_value">CGST
                                        </label>
                                    </div>
                                    <input type="text" name="cgst_value" id="cgst_value" value="{{ cgst_value }}" class="form-control">
                                </div>
                            </div>
                            <div class="col-lg-6">
                                <div class="input-group input-group-sm mb-3">
                                    <div class="input-group-prepend">
                                        <label class="input-group-text" for="invoice_value">SGST
                                        </label>
                                    </div>
                                    <input type="text" name="sgst_value" id="sgst_value" value="{{ sgst_value }}" class="form-control">
                                </div>
                            </div>

                        </div>

                        <div class="row">



                            <div class="col-lg-4"></div>
                            <!-- </div> -->
                                <div class="card-body" id="table1">
                                    <div class="container">

                                        <table id="myTable1" class=" table order-list table table-responsive-md"
                                            style="table-layout:fixed;width:850px;">

                                            <col width="200">
                                            <col width="200">
                                            <col width="200">
                                            <col width="200">
                                            <col width="200">
                                            <col width="200">
                                            <col width="150">
                                            <thead class="thead-dark" style="border-color: #353535;">
                                                <tr>
                                                    <!-- <th class="headcol" style="margin-left: -26px;" bgcolor="#E6E6FA">Action</th> -->
                                                    <th scope="col" class="text-center fontSizeOfth">Item Description</th>
                                                    <th scope="col" class="text-center fontSizeOfth">HSN Code</th>
                                                    <th scope="col" class="text-center fontSizeOfth">Quantity</th>
                                                    <th scope="col" class="text-center fontSizeOfth">Taxable Amount</th>
                                                    <th scope="col" class="text-center fontSizeOfth">GST Rate</th>
                                                    <th scope="col" class="text-center fontSizeOfth">Total Amount</th>
                                                </tr>
                                            </thead>
                                            <tbody class="table-bordered" style="overflow-y:auto;height:200px;padding-left:10%;">
                                                <tr>
                                                    <td class="pt-3-half">
                                                        <input type="text" name="item_desc[]" value="" id="gstrate" style="width:105%; height:20%">
                                                    </td>
                                                    <td class="pt-3-half">
                                                        <input type="text" name="hsn_code[]" value="" id="gstrate" style="width:105%; height:20%">
                                                    </td>
                                                    <td class="pt-3-half">
                                                        <input type="text" name="quantity[]" value="" id="gstrate" style="width:105%; height:20%">
                                                    </td>
                                                    <td class="pt-3-half">
                                                        <input type="text" name="taxable_value[]" value="" id="gstrate" style="width:105%; height:20%">
                                                    </td>
                                                    <td class="pt-3-half">
                                                        <input type="text" name="gst_rate[]" value="" id="totalgst" style="width:105%; height:20%">
                                                    </td>
                                                    <td class="pt-3-half">
                                                        <input type="text" name="total_amount[]" value="" id="gstrate" style="width:105%; height:20%">
                                                    </td>
                                                </tr>
                                            </tbody>
                                            <tfoot>
                                                <tr>
                                                </tr>
                                            </tfoot>
                                        </table>
                                    </div>
                                </div>
                        </div>
                        <!-- Table Close -->
                    </div>
                    <div style="padding-top: 0%;">
                        <div style="padding-top: 0%;" id="submit1">
                            <button type="submit" style="margin-bottom: 5px; color: white;"
                                class="btn btn-dark">Submit</button>
                        </div>
                    </div>
            </form>
        </div>

    </div>
</div>

<script>
    $('#ship_to').click(function () {
        if ($("#ship_to").prop("checked") == false) {
            $("#Ven_name").attr("hidden", true);
            $("#POSState").attr("hidden", true);
            $("#POSCountry").attr("hidden", true);
            $("#gst_no_of_bill_to").attr("hidden", true);
        }
        if ($("#ship_to").prop("checked") == true) {
            console.log("check")
            $("#Ven_name").attr("hidden", false);
            $("#POSState").attr("hidden", false);
            $("#POSCountry").attr("hidden", false);
            $("#gst_no_of_bill_to").attr("hidden", false)
            $("#state112").on("change", function () {
                console.log("vendorchange")
                var firstvendor = $("#vendor_name").val()
                console.log(firstvendor)
                var secondvendor = $("#vendor_name_of_ship_to").val()
                console.log(secondvendor)
                var gstnooffirstvendor = $("#gstnoofshipto").val()
                console.log(gstnooffirstvendor)
                var gstoffirstvendor = gstnooffirstvendor.slice(0, 2)
                console.log(gstoffirstvendor)
                var gstnoofsecondvendor = $("#gst_no_of_bill_to_ship_to").val()
                console.log(gstnoofsecondvendor)
                var gstofsecondvendor = gstnoofsecondvendor.slice(0, 2)
                console.log(gstofsecondvendor)
                if (firstvendor == secondvendor) {
                    console.log("samevendor")
                    if (gstoffirstvendor == gstofsecondvendor) {
                        console.log("samegst")
                        console.log($("#igst").val())
                        if ($("#igst").val() != "") {
                            console.log("notblank")
                            console.log($("#whether_gst_credit_available").val())
                            $("#whether_gst_credit_available").prop("checked", false);
                        }
                    }
                    else {
                        if ($("#cgst").val() != "" && $("#sgst").val() != "") {
                            $("#whether_gst_credit_available").prop("checked", false);
                        }
                    }
                }

            });
        }
    });
    function myFunction1() {
        $("#old_reference").attr("hidden", false);
    }
    $(document).ready(function () {
        $("#click1").attr("hidden", true);
        $("#click2").attr("hidden", true);
        $("#click3").attr("hidden", true);
        $("#click4").attr("hidden", true);

    }); function myFunction() {
        $("#old_reference10").attr("hidden", false);
    }
    $(function () {
        var sel = $('#vendor_name');
        var selected = sel.val(); // cache selected value, before reordering
        var opts_list = sel.find('option');
        opts_list.sort(function (a, b) { return $(a).text() > $(b).text() ? 1 : -1; });
        sel.html('').append(opts_list);
        sel.val(selected);
    });
    $(function () {
        var sel = $('#voucher_type');
        var selected = sel.val(); // cache selected value, before reordering
        var opts_list = sel.find('option');
        opts_list.sort(function (a, b) { return $(a).text() > $(b).text() ? 1 : -1; });
        sel.html('').append(opts_list);
        sel.val(selected);
    });
    $(function () {
        var sel = $('#reference_type');
        var selected = sel.val(); // cache selected value, before reordering
        var opts_list = sel.find('option');
        opts_list.sort(function (a, b) { return $(a).text() > $(b).text() ? 1 : -1; });
        sel.html('').append(opts_list);
        sel.val(selected);
    });
    $(function () {
        var sel = $('#payment_duty');
        var selected = sel.val(); // cache selected value, before reordering
        var opts_list = sel.find('option');
        opts_list.sort(function (a, b) { return $(a).text() > $(b).text() ? 1 : -1; });
        sel.html('').append(opts_list);
        sel.val(selected);
    });
    $(function () {
        var sel = $('#suply_state2');
        var selected = sel.val(); // cache selected value, before reordering
        var opts_list = sel.find('option');
        opts_list.sort(function (a, b) { return $(a).text() > $(b).text() ? 1 : -1; });
        sel.html('').append(opts_list);
        sel.val(selected);
    });
    $(function () {
        var sel = $('#suply_country');
        var selected = sel.val(); // cache selected value, before reordering
        var opts_list = sel.find('option');
        opts_list.sort(function (a, b) { return $(a).text() > $(b).text() ? 1 : -1; });
        sel.html('').append(opts_list);
        sel.val(selected);
    });
    $(function () {
        var sel = $('#expense_head');
        var selected = sel.val(); // cache selected value, before reordering
        var opts_list = sel.find('option');
        opts_list.sort(function (a, b) { return $(a).text() > $(b).text() ? 1 : -1; });
        sel.html('').append(opts_list);
        sel.val(selected);
    });
    $(function () {
        var sel = $('#expense_head1');
        var selected = sel.val(); // cache selected value, before reordering
        var opts_list = sel.find('option');
        opts_list.sort(function (a, b) { return $(a).text() > $(b).text() ? 1 : -1; });
        sel.html('').append(opts_list);
        sel.val(selected);
    });
    $(function () {
        var sel = $('#expense_head3');
        var selected = sel.val(); // cache selected value, before reordering
        var opts_list = sel.find('option');
        opts_list.sort(function (a, b) { return $(a).text() > $(b).text() ? 1 : -1; });
        sel.html('').append(opts_list);
        sel.val(selected);
    });
    $(function () {
        var sel = $('#expense_head4');
        var selected = sel.val(); // cache selected value, before reordering
        var opts_list = sel.find('option');
        opts_list.sort(function (a, b) { return $(a).text() > $(b).text() ? 1 : -1; });
        sel.html('').append(opts_list);
        sel.val(selected);
    });
    $(function () {
        var sel = $('#uqc');
        var selected = sel.val(); // cache selected value, before reordering
        var opts_list = sel.find('option');
        opts_list.sort(function (a, b) { return $(a).text() > $(b).text() ? 1 : -1; });
        sel.html('').append(opts_list);
        sel.val(selected);
    });
    $(function () {
        var sel = $('#uqc_type');
        var selected = sel.val(); // cache selected value, before reordering
        var opts_list = sel.find('option');
        opts_list.sort(function (a, b) { return $(a).text() > $(b).text() ? 1 : -1; });
        sel.html('').append(opts_list);
        sel.val(selected);
    });
    $(function () {
        var sel = $('#reference_type12');
        var selected = sel.val(); // cache selected value, before reordering
        var opts_list = sel.find('option');
        opts_list.sort(function (a, b) { return $(a).text() > $(b).text() ? 1 : -1; });
        sel.html('').append(opts_list);
        sel.val(selected);
    });
    $(function () {
        var sel = $('#reference');
        var selected = sel.val(); // cache selected value, before reordering
        var opts_list = sel.find('option');
        opts_list.sort(function (a, b) { return $(a).text() > $(b).text() ? 1 : -1; });
        sel.html('').append(opts_list);
        sel.val(selected);
    });
    $(function () {
        var sel = $('#goods_or_services');
        var selected = sel.val(); // cache selected value, before reordering
        var opts_list = sel.find('option');
        opts_list.sort(function (a, b) { return $(a).text() > $(b).text() ? 1 : -1; });
        sel.html('').append(opts_list);
        sel.val(selected);
    });
    $(function () {
        var sel = $('#supply_state');
        var selected = sel.val(); // cache selected value, before reordering
        var opts_list = sel.find('option');
        opts_list.sort(function (a, b) { return $(a).text() > $(b).text() ? 1 : -1; });
        sel.html('').append(opts_list);
        sel.val(selected);
    });
    $(function () {
        var sel = $('#foreign_currency');
        var selected = sel.val(); // cache selected value, before reordering
        var opts_list = sel.find('option');
        opts_list.sort(function (a, b) { return $(a).text() > $(b).text() ? 1 : -1; });
        sel.html('').append(opts_list);
        sel.val(selected);
    });
    function hide(abc) {

        if (abc == 1) {

            $("#old_reference").attr("hidden", true);

        }

    }
    function hide1(abc) {

        if (abc == 1) {

            $("#old_reference10").attr("hidden", true);

        }

    }
    $("#discount").on("change", function () {
        var dis = $("#discount").val()
        $("#discount1").val(dis);
    });
    $("#payment_duty").on("change", function () {
        console.log("change")
        if ($("#payment_duty").val() == "Yes") {
            $("#igst10").attr("hidden", false);
        }
        else {
            console.log("No")
            $("#igst10").attr("hidden", true);
        }

    });
    $("#vendor_name").on("change", function () {
        var ledger_name = $('#vendor_name').val();
        console.log("billbybill")
        $.ajax({
            url: '../../billbybill/' + "{{company_code}}",
            data: {
                ledger_name: ledger_name,
                csrfmiddlewaretoken: '{{ csrf_token }}',
            },
            dataType: 'json',
            type: 'POST',
            success: function (data) {
                //alert(data);
                console.log(data); //This is pure text.
                console.log(data.result);
                // condole.log(data.result)
                var billbybill = data.result;
                if (billbybill[0][0] == "Yes") {
                    $("#reference_type1").attr("hidden", false);
                    $("#reference_type11").attr("hidden", false);
                    $("#reference_type2").attr("hidden", false);
                }
                else {
                    $("#reference_type1").attr("hidden", true)
                    $("#reference_type11").attr("hidden", true)
                    $("#reference_type2").attr("hidden", true)
                }
                if (billbybill[0][1] == "Nominal") {
                    $(".cost_center10").attr("hidden", false);
                    console.log("if")
                }
                else {
                    console.log("else")
                    $(".cost_center10").attr("hidden", true);

                }
                $("#gstnoofshipto").val(billbybill[0][2])

            }
        });

    });

    var i = parseFloat("{% if x %}{{ x }}{% else %}0{% endif %}");
    $('#if_items').click(function () {
        if ($("#if_items").prop("checked") == false) {
            $("#table-ocr").hide();
            $("#addrow").hide();
            $("#table1").show()
            $("#addrow1").show();

        }
        if ($("#if_items").prop("checked") == true) {
            $("#table-ocr").show();
            $("#addrow").show();
            $("#table1").hide();
            $("#addrow1 ").hide();

        }
    });

    $("#igst3").on("change", function () {
        var igst = $("#igst3").val();
        $("#totalgst3").val(igst);

    });
    $("#igst3").on("change", function () {
        var igst = $("#igst3").val();
        $("#totalgst5").val(igst);

    });
    $('#gstcredit').click(function () {
        if ($("#gstcredit").prop("checked") == false) {
            var amount = parseFloat($("#amount3").val());
            var amount1 = parseFloat($("#amount5").val())
            var gst = parseFloat($("#totalgst3").val());
            var gst1 = parseFloat($("#totalgst5").val());
            var totalamount = amount + gst;
            var totalamount1 = amount1 + gst1
            $("#amount3").val(totalamount1);
            $("#totalgst3").val("");
        }
        else {
            $("#amount3").val($("#bcd1").val());
            $("#amount5").val($("#bcd1").val());
            $("#totalgst3").val($("#igst3").val());
            $("#totalgst5").val($("#igst3").val());
        }
    });
    $('#if_items1').click(function () {
        console.log("if_items")
        if ($("#if_items1").prop("checked") == false) {

            $("#table-ocr1").hide();
            $("#addrow3").hide();
            $("#table2").show()
            $("#addrow2").show();

        }
        if ($("#if_items1").prop("checked") == true) {
            $("#table-ocr1").show();
            $("#addrow3").show();
            $("#table2").hide();
            $("#addrow2 ").hide();

        }
    });

    $('#if_invoices').click(function () {
        console.log("ggggggggggg")
        if ($("#if_invoices").prop("checked") == false) {
            console.log("ggggggggggg1")
            $("#invoice_no112").hide();
            $("#invoice_no121").hide();
        }
        else {
            console.log("ggggggggggg2")
            $("#invoice_no112").show();
            $("#invoice_no121").show();
        }
    });

    $("#bcd1").on("change", function () {
        var amount = $("#bcd1").val();
        $("#amount3").val(amount)
        $("#amount5").val(amount)
    });
    $("#invoice_value2").on("change", function () {
        var exchangerate = $("#exchange_rate2").val()
        if (isNaN(exchangerate)) {
            var amount = $("#invoice_value2").val();
        }
        else {
            var amount = $("#invoice_value2").val() * $("#exchange_rate2").val();
        }
        $("#amount2").val(amount)
        $("#amount4").val(amount)

    });
    $("#exchange_rate2").on("change", function () {
        console.log("exchange_rate")
        var exchangerate = $("#exchange_rate2").val()
        console.log(exchangerate)
        if (isNaN(exchangerate)) {
            var amount = $("#invoice_value2").val();
            console.log(amount)
        }
        else {
            var amount = $("#invoice_value2").val() * $("#exchange_rate2").val();
        }
        $("#amount2").val(amount)
        $("#amount4").val(amount)

    });
    $(document).ready(function () {
        if ($("#voucher_type").val() == "Sales") {
            console.log("purchase")
            $("#submit2").attr("hidden", false);
            $("#submit1").attr("hidden", true);
        }
    });
    $("#voucher_type").on("change", function () {
        if ($("#voucher_type").val() == "Purchase" || $("#voucher_type").val() == "Import" || $("#voucher_type").val() == "Export" || $("#voucher_type").val() == "SEZ") {
            console.log("purchase")
            $("#submit2").attr("hidden", true);
            $("#submit1").attr("hidden", false);
        } else {
            console.log("sales")
            $("#submit2").attr("hidden", false);
            $("#submit1").attr("hidden", true);
        }
        if ($("#voucher_type").val() == "Import" || $("#voucher_type").val() == "Export" || $("#voucher_type").val() == "SEZ") {
            (/, console.log("purchase"))
            $("#other").attr("hidden", true);
            $("#import").attr("hidden", false);
        } else {
            $("#other").attr("hidden", false);
            $("#import").attr("hidden", true);
        }

        if ($("#voucher_type").val() == "SEZ") {
            $("#shipping_bill_no").attr("hidden", true);
            $("#shipping_bill_date_date1121").attr("hidden", true);
            $("#bcd").attr("hidden", true);
            $("#exchange_rate1").attr("hidden", true);
        } else {
            $("#shipping_bill_no").attr("hidden", false);
            $("#shipping_bill_date_date1121").attr("hidden", false);
            $("#bcd").attr("hidden", false);
            $("#exchange_rate1").attr("hidden", false);
        }

    });
    $("#expense_head1").on("change", function () {
        var gstrate1 = parseFloat($("#gstrate").val());
        var invoicevalue1 = parseFloat($("#invoice_value").val());
        var totalgst1 = parseFloat($("#totalgst").val());
        if (invoicevalue1 != totalgst1) {
            var newRow = $("<tr>");
            var cols = "";
            cols +=
                '<td><input type="button" class="ibtnDel btn btn-md btn-sm btn-danger headcol" style=height:2.5em; value="Delete" bgcolor="#E6E6FA"></td>';
            "{% if vouchertype == '1' or vouchertype == '2' %}"
            cols += '<td class="pt-3-half" ' + counter +
                '"><select  style="word-wrap:break-word;width:100%;" class="expense" id="expense_head' + counter +
                '" name = "expense_type3">{% for element in ledgers %}<option value="{{element.ledger_name}}">{{element.ledger_name}}</option>{% endfor %}</td>';
            cols += '<td class="pt-3-half" ' + counter +
                '"><input type="text" name = "taxablevalue" style = "width:105%; height:20%"> </td>';
            "{% if vouchertype == '2' %}"
            cols += '<td class="pt-3-half" ' + counter +
                '"><input type="text" name = "discount" style = "width:105%; height:20%"> </td>';
            "{% endif %}"
            cols += '<td class="pt-3-half" ' + counter +
                '"><input type="text" name = "gst_name" id="gst' + counter +
                '" style = "width:115%; height:20%"></td>';
            cols += '<td class="pt-3-half" ' + counter +
                '"><input type="text" name = "totalgst" id="gstrate ' + counter +
                '" style = "width:115%; height:20%"></td>';
            cols += '<td class="pt-3-half cost_center10" ' + counter +
                '"><select style="word-wrap:break-word;width:100%;" id="cost_center' + counter +
                '" name = "cost_center3">{% for element in cost_center_info %}<option value="{{element.cost_center_name}}">{{element.cost_center_name}}</option>{% endfor %}</td>';
            cols += '<td class="pt-3-half" ' + counter +
                '"><input type="text" name = "hsn" id="hsn' + counter +
                '" style = "width:115%; height:20%"></td>';
            cols += '<td class="pt-3-half" ' + counter +
                '"><input type="text" name = "itemdescription" id="itemdescription' + counter +
                '" class = "item_service_ddescription" ' + counter + ' style = "width:105%; height:20%"></td>';
            cols += '<td class="pt-3-half" ' + counter +
                '"><select style="word-wrap:break-word;width:100%;" id="uqc' + counter +
                '" name = "uqc_type">{% for element in uqc_type %}<option value="{{element.uqc}}">{{element.uqc}}</option>{% endfor %}</td>';

            cols += '<td class="pt-3-half" ' + counter +
                '"><input type="text" name = "qty" style = "width:115%; height:20%"></td>';


            cols += '<td class="pt-3-half" ' + counter +
                ' "> <select style="word-wrap:break-word;width:100%;" name = "goods_or_services2">{% for element in goods_or_services %}<option value="{{element.goods_or_services}}">{{element.goods_or_services}}</option>{% endfor %}</td>';
            "{% endif %}"
            "{% if vouchertype == '1' %}"
            cols += '<td class="pt-3-half" ' + counter +
                '"><input type="date" class="startDate" id="startdate' + counter +
                '" name = "startdate" style = "width:110%; height:20%" readonly> </td>';
            cols += '<td class="pt-3-half" ' + counter +
                '"><input type="date" class="endDate" id="enddate' + counter +
                '" name = "enddate" style = "width:110%; height:20%" readonly></td>';
            "{% endif %}"

            "{% if vouchertype == '3' or vouchertype == '4' or vouchertype == '5' or vouchertype == '6' %}"
            cols += '<td class="pt-3-half"  ' + counter +
                '" > <select  style="word-wrap:break-word;width:100%;" id="reference' + counter +
                '">{% for element in reference %}<option value="{{element.ref_type}}" >{{element.ref_type}}</option>{% endfor %}</select></td>';
            cols += '<td class="pt-3-half" ' + counter +
                '"><input type = "text"  style = "width:100%" name = "amount" id = "amount' + counter +
                '" > </td>';
            cols += '<td class="pt-3-half"   ' + counter +
                ' "> <input type = "text" id = "old_reference' + counter +
                '" name = "old_reference" style = "width:100%"  readonly> </td>';
            "{% endif %}"

            console.log(counter)

            newRow.append(cols);
            $("#myTable").append(newRow);
            counter++;
            $("#expense").select2();
            "{% if vouchertype == '3' or vouchertype == '4' or vouchertype == '5' or vouchertype == '6' %}"
            $("#reference" + (counter - 1)).on("change", function () {
                console.log("hello")
                if ($("#reference" + (counter - 1)).val() == "Agst Ref") {
                    console.log("You chose agst ref");
                    $("#old_reference" + (counter - 1)).attr("readonly", false);
                } else {
                    $("#old_reference" + (counter - 1)).attr("readonly", true);

                }
            });
            "{% endif %}"

            $(document).ready(function () {

                console.log(counter - 1)

                $("#goods_or_services" + (counter - 1)).on("change", function () {
                    console.log("chahal")

                    if ($("#goods_or_services" + (counter - 1)).val() == "G") {
                        $("#startdate" + (counter - 1)).attr('readonly', true)
                        $("#enddate" + (counter - 1)).attr('readonly', true)
                        $("#qty" + (counter - 1)).attr('readonly', false)
                        $("#uqc_type" + (counter - 1)).attr('readonly', false)
                    } else {
                        $("#startdate" + (counter - 1)).attr('readonly', false)
                        $("#enddate" + (counter - 1)).attr('readonly', false)
                        $("#qty" + (counter - 1)).attr('readonly', true)
                        $("#uqc_type" + (counter - 1)).attr('readonly', true)
                    }
                });


            });

            var date = new Date();
            var day = date.getDate();
            var month = date.getMonth() + 1;
            var year = date.getFullYear();
            if (month < 10) month = "0" + month;
            if (day < 10) day = "0" + day;
            var today = year + "-" + month + "-" + day;
            (/, document.getElementById('startdate', +, (counter, -, 1)).value, =, today;)
            (/, $(".startDate").val(today);)
            (/, document.getElementById('enddate', +, (counter, -, 1)).value, =, today;)
            (/, $(".endDate").val(today))





            $("#myTable").on("click", ".ibtnDel", function (event) {
                $(this).closest("tr").remove();
                counter -= 1
            });
        }


        if (gstrate1 != 0 || gstrate1 != 4 || gstrate1 != 12 || gstrate1 != 18) {
            $('#modal').modal('toggle');

        } else {
            $('#modal').modal('hide');

        }


    });
    $("#expense_head").on("change", function () {
        var gstrate = parseFloat($("#gstrate").val());
        var invoicevalue = parseFloat($("#invoice_value").val());
        var totalgst = parseFloat($("#totalgst").val());
        if (invoicevalue != totalgst) {
            counter = 0;
            var newRow = $("<tr>");
            var cols = "";
            cols +=
                '<td><input type="button" class="ibtnDel btn btn-md btn-sm btn-danger headcol" style=height:2.5em; value="Delete" bgcolor="#E6E6FA"></td>';
            "{% if vouchertype == '1' or vouchertype == '2' %}"
            cols += '<td class="pt-3-half" ' + counter +
                '"><select style="word-wrap:break-word;width:100%;" id="expense_head" name = "expense_type">{% for element in ledgers %}<option value="{{element.ledger_name}}">{{element.ledger_name}}</option>{% endfor %}</td>';
            cols += '<td class="pt-3-half" ' + counter +
                '"><input type="text" name = "taxablevalue3" style = "width:105%; height:20%"> </td>';
            cols += '<td class="pt-3-half" ' + counter +
                '"><input type="text" name = "gst_name4" style = "width:105%; height:20%"> </td>';
            cols += '<td class="pt-3-half" ' + counter +
                '"><input type="text" name = "totalgst8" id="gst' + counter +
                '" style = "width:115%; height:20%"></td>';
            cols += '<td class="pt-3-half cost_center10" ' + counter +
                '"><select style="word-wrap:break-word;width:100%;" name = "cost_center8">{% for element in cost_center_info %}<option value="{{element.cost_center_name}}">{{element.cost_center_name}}</option>{% endfor %}</td>';
            console.log(counter)
            "{% endif %}"
            "{% if vouchertype == '3' or vouchertype == '4' or vouchertype == '5' or vouchertype == '6' %}"
            cols += '<td class="pt-3-half" ' + counter +
                '"><select style="word-wrap:break-word;width:100%;" id="expense_head" name = "expense_type">{% for element in ledgers %}<option value="{{element.ledger_name}}">{{element.ledger_name}}</option>{% endfor %}</td>';

            cols += '<td class="pt-3-half" ' + counter +
                '"><input type = "text"  style = "width:100%" name = "amount" id = "amount' +
                counter + '" > </td>';
            cols += '<td class="pt-3-half"  ' + counter +
                '" > <select  style="word-wrap:break-word;width:100%;" id="reference' +
                counter +
                '">{% for element in reference %}<option value="{{element.ref_type}}" >{{element.ref_type}}</option>{% endfor %}</select></td>';

            cols += '<td class="pt-3-half"   ' + counter +
                ' "> <input type = "text" id = "old_reference' + counter +
                '" name = "old_reference" style = "width:100%"  readonly> </td>';
            cols += '<td class="pt-3-half cost_center10" ' + counter +
                '" ><select style="word-wrap:break-word;width:100%;" name = "cost_center8">{% for element in cost_center_info %}<option value="{{element.cost_center_name}}">{{element.cost_center_name}}</option>{% endfor %}</td>';

            "{% endif %}"
            newRow.append(cols);
            $("#myTable1").append(newRow);
            counter++;

        }

        $("#myTable1").on("click", ".ibtnDel", function (event) {
            $(this).closest("tr").remove();
            counter -= 1







        });

        if (gstrate != 0 || gstrate != 4 || gstrate != 12 || gstrate != 18) {
            $('#modal').modal('toggle');

        } else {
            $('#modal').modal('hide');

        }


    });
    $("#reference_type").on("change", function () {
        console.log("referencetypeagst")

        if ($("#reference_type").val() == "Agst Ref") {
            $("#agst").attr("hidden", true);
            $("#agst1").attr("hidden", true);
            $("#row1").attr("hidden", true);
            $("#row2").attr("hidden", true);
            $("#click1").attr("hidden", true);
            $("#click2").attr("hidden", true);
            $("#click3").attr("hidden", true);
            $("#click4").attr("hidden", true);
        }
        else {
            $("#click1").attr("hidden", false);
            $("#click2").attr("hidden", false);
            $("#click3").attr("hidden", false);
            $("#click4").attr("hidden", false);
            $("#agst").attr("hidden", false);
            $("#agst1").attr("hidden", false);
            $("#row1").attr("hidden", false);
            $("#row2").attr("hidden", false);
        }

    });
    $("#reference").on("change", function () {
        if ($("#reference").val() == "Agst Ref" && !$("#vendor_name").val()) {
            alert("Please choose a valid Vendor Name.");
            $("#old_reference10").attr("hidden", true);
            $("#reference").val("New Ref");
        } else if ($("#reference").val() == "Agst Ref" && $("#vendor_name").val()) {
            var vendor_name = $("#vendor_name").val();
            $.ajax({
                url: '../../get-old-ref/',
                data: {
                    vendor_name: vendor_name,
                    csrfmiddlewaretoken: '{{ csrf_token }}',
                },
                dataType: 'json',
                type: 'POST',

                success: function (response) {
                    // $("#old_ref_list").empty();
                    console.log(response);
                    $('#Table20').empty();
                    var trHTML = '';
                    $.each(response, function (i, item) {
                        trHTML += '<tr><td><input type="checkbox" name="check_list[]" value="' + item[1] + '_' + item[4] + '"></td><td><input name="text1" value="' + item[1] + '" readonly></td><td><input  name="text2" value="' + item[2] + '" readonly></td><td><input  name="text3" value="' + item[3] + '" readonly></td><td><input  name="text4" value="' + item[4] + '" readonly></td></tr>';
                    });
                    $('#Table20').append(trHTML);

                },


                error: function (data) {
                    console.log("ERROR!!!");
                }
            });
            $("#old_reference10").attr("hidden", false);
            $("#click").attr("hidden", false);
        } else {
            $("#old_reference10").attr("hidden", true);
        }
    });
    $("#reference_type11").on("change", function () {
        console.log("referencetypeagst")

        if ($("#reference_type11").val() == "Agst Ref") {
            console.log("referencetype")
            $("#agst").attr("hidden", true);
            $("#agst1").attr("hidden", true);
            $("#row1").attr("hidden", true);
            $("#row2").attr("hidden", true);
        }
        else {
            $("#agst").attr("hidden", false);
            $("#agst1").attr("hidden", false);
            $("#row1").attr("hidden", false);
            $("#row2").attr("hidden", false);
        }

    });
    $("#reference_type").on("change", function () {
        if ($("#reference_type").val() == "Agst Ref" && !$("#vendor_name").val()) {
            alert("Please choose a valid Vendor Name.");
            $("#old_reference").attr("hidden", true);
            $("#reference_type").val("New Ref");
        } else if ($("#reference_type").val() == "Agst Ref" && $("#vendor_name").val()) {
            var vendor_name = $("#vendor_name").val();
            $.ajax({
                url: '../../get-old-ref/',
                data: {
                    vendor_name: vendor_name,
                    csrfmiddlewaretoken: '{{ csrf_token }}',
                },
                dataType: 'json',
                type: 'POST',

                success: function (response) {
                    // $("#old_ref_list").empty();
                    console.log(response);
                    $('#Table').empty();
                    var trHTML = '';
                    $.each(response, function (i, item) {
                        trHTML += '<tr><td><input type="checkbox" name="check_list[]" value="' + item[1] + '_' + item[4] + '"></td><td><input name="text1" value="' + item[1] + '" readonly></td><td><input  name="text2" value="' + item[2] + '" readonly></td><td><input  name="text3" value="' + item[3] + '" readonly></td><td><input  name="text4" value="' + item[4] + '" readonly></td></tr>';
                    });
                    $('#Table').append(trHTML);

                },


                error: function (data) {
                    console.log("ERROR!!!");
                }
            });
            $("#old_reference").attr("hidden", false);
            $("#click").attr("hidden", false);
        } else {
            $("#old_reference").attr("hidden", true);
        }
    });
    $("#vendor_name").on("change", function () {
        if ($("#reference_type").val() == "Agst Ref" && $("#vendor_name").val()) {
            var vendor_name = $("#vendor_name").val();
            $.ajax({
                url: '../../get-old-ref/',
                data: {
                    vendor_name: vendor_name,
                    csrfmiddlewaretoken: '{{ csrf_token }}',
                },
                dataType: 'json',
                type: 'POST',

                success: function (response) {
                    // $("#old_ref_list").empty();
                    console.log(response);
                    $('#Table').empty();
                    var trHTML = '';
                    $.each(response, function (i, item) {
                        trHTML += '<tr><td><input type="checkbox" name="check_list[]" value="' + item[1] + '_' + item[4] + '"></td><td><input name="text1" value="' + item[1] + '" readonly></td><td><input  name="text2" value="' + item[2] + '" readonly></td><td><input  name="text3" value="' + item[3] + '" readonly></td><td><input  name="text4" value="' + item[4] + '" readonly></td></tr>';
                    });
                    $('#Table').append(trHTML);

                },


                error: function (data) {
                    console.log("ERROR!!!");
                }
            });
            $("#old_reference").attr("hidden", false);
        } else {
            $("#old_reference").attr("hidden", true);
        }
    });

    $("#reference_type12").on("change", function () {
        if ($("#reference_type12").val() == "Agst Ref" && !$("#vendor_name").val()) {
            alert("Please choose a valid Vendor Name.");
            $("#old_reference10").attr("hidden", true);
            $("#reference_type12").val("New Ref");
        } else if ($("#reference_type12").val() == "Agst Ref" && $("#vendor_name").val()) {
            var vendor_name = $("#vendor_name").val();
            $.ajax({
                url: '../../get-old-ref/',
                data: {
                    vendor_name: vendor_name,
                    csrfmiddlewaretoken: '{{ csrf_token }}',
                },
                dataType: 'json',
                type: 'POST',

                success: function (response) {
                    // $("#old_ref_list").empty();
                    console.log(response);
                    $('#Table10').empty();
                    var trHTML = '';
                    $.each(response, function (i, item) {
                        trHTML += '<tr><td><input type="checkbox" name="check_list[]" value="' + item[1] + '_' + item[4] + '"></td><td><input name="text1" value="' + item[1] + '" readonly></td><td><input  name="text2" value="' + item[2] + '" readonly></td><td><input  name="text3" value="' + item[3] + '" readonly></td><td><input  name="text4" value="' + item[4] + '" readonly></td></tr>';
                    });
                    $('#Table10').append(trHTML);

                },


                error: function (data) {
                    console.log("ERROR!!!");
                }
            });
            $("#old_reference10").attr("hidden", false);
            $("#click1").attr("hidden", false);
            $("#click2").attr("hidden", false);
            $("#click3").attr("hidden", false);
            $("#click4").attr("hidden", false);
        } else {
            $("#old_reference10").attr("hidden", true);
            $("#click1").attr("hidden", true);
            $("#click2").attr("hidden", true);
            $("#clic3").attr("hidden", true);
            $("#clic4").attr("hidden", true);

        }
    });

    $("#vendor_name").on("change", function () {
        if ($("#reference_type12").val() == "Agst Ref" && $("#vendor_name").val()) {
            var vendor_name = $("#vendor_name").val();
            $.ajax({
                url: '../../get-old-ref/',
                data: {
                    vendor_name: vendor_name,
                    csrfmiddlewaretoken: '{{ csrf_token }}',
                },
                dataType: 'json',
                type: 'POST',

                success: function (response) {
                    // $("#old_ref_list").empty();
                    console.log(response);
                    $('#Table10').empty();
                    var trHTML = '';
                    $.each(response, function (i, item) {
                        trHTML += '<tr><td><input type="checkbox" name="check_list[]" value="' + item[1] + '_' + item[4] + '"></td><td><input name="text1" value="' + item[1] + '" readonly></td><td><input  name="text2" value="' + item[2] + '" readonly></td><td><input  name="text3" value="' + item[3] + '" readonly></td><td><input  name="text4" value="' + item[4] + '" readonly></td></tr>';
                    });
                    $('#Table10').append(trHTML);

                },


                error: function (data) {
                    console.log("ERROR!!!");
                }
            });
            $("#old_reference10").attr("hidden", false);
        } else {
            $("#old_reference10").attr("hidden", true);
        }
    });

    $("#vendor_name").on("change", function () {
        if ($("#reference").val() == "Agst Ref" && $("#vendor_name").val()) {
            var vendor_name = $("#vendor_name").val();
            $.ajax({
                url: '../../get-old-ref/',
                data: {
                    vendor_name: vendor_name,
                    csrfmiddlewaretoken: '{{ csrf_token }}',
                },
                dataType: 'json',
                type: 'POST',

                success: function (response) {
                    // $("#old_ref_list").empty();
                    console.log(response);
                    $('#Table20').empty();
                    var trHTML = '';
                    $.each(response, function (i, item) {
                        trHTML += '<tr><td><input type="checkbox" name="check_list[]" value="' + item[1] + '_' + item[4] + '"></td><td><input name="text1" value="' + item[1] + '" readonly></td><td><input  name="text2" value="' + item[2] + '" readonly></td><td><input  name="text3" value="' + item[3] + '" readonly></td><td><input  name="text4" value="' + item[4] + '" readonly></td></tr>';
                    });
                    $('#Table20').append(trHTML);

                },


                error: function (data) {
                    console.log("ERROR!!!");
                }
            });
            $("#old_reference10").attr("hidden", false);
        } else {
            $("#old_reference10").attr("hidden", true);
        }
    });

    $(document).ready(function () {
        "{% for key,value in results.items %}"

        $(".{{key}}").val("{{value}}")

        "{% endfor %}"
    });
    $(document).ready(function () {
        $(".vendor_name").select2();
        $(".supply_state").select2();
        $("#expense_head").select2();
        $("#expense_head2").select2();
        $("#expense_hea4").select2();
        $("#expense_hea44").select2();
        $("#expense_head1").select2();
        $("#expense_head3").select2();
        $(".expense55").select2();
        $(".expense21").select2();
        $(".expense22").select2();
        (/, $(".expense23").select2();)
        (/, $(".expense57").select2();)
        $(".expense58").select2();
        $(".expense59").select2();
        $("#reference").select2();
        $("#reference_type").select2();
        $("#reference_type12").select2();
        $("#uqc").select2();
        $("#uqc1").select2();
        $("#uqc_type").select2();
        $("#goods_or_services").select2();
        $("#goods_or_services1").select2();
        $("#goods_or_services2").select2();
        $("#cc").select2();
        $("#cc1").select2();
        $("#cc2").select2();
        $("#cc3").select2();
        $("#cc4").select2();
        $("#cc5").select2();
        $("#c6").select2();
        $("#cc7").select2();


        var counter = 0;
        $("#addrow").on("click", function () {
            var newRow = $("<tr>");
            var cols = "";
            cols +=
                '<td><input type="button" class="ibtnDel btn btn-md btn-sm btn-danger headcol" style=height:2.5em; value="Delete" bgcolor="#E6E6FA"></td>';
            "{% if vouchertype == '1' or vouchertype == '2' %}"
            cols += '<td class="pt-3-half" ' + counter +
                '"><select  style="word-wrap:break-word;width:100%;" class="expense_items" id="expense_head' + counter +
                '" name = "expense_type3">{% for element in ledgers %}<option value="{{element.ledger_name}}">{{element.ledger_name}}</option>{% endfor %}</td>';
            cols += '<td class="pt-3-half" ' + counter +
                '"><input type="text" name = "taxablevalue" style = "width:105%; height:20%"> </td>';
            "{% if vouchertype == '2' %}"
            cols += '<td class="pt-3-half" ' + counter +
                '"><input type="text" name = "discount" style = "width:105%; height:20%"> </td>';
            "{% endif %}"
            cols += '<td class="pt-3-half" ' + counter +
                '"><input type="text" name = "gst_name" id="gst' + counter +
                '" style = "width:115%; height:20%"></td>';
            cols += '<td class="pt-3-half" ' + counter +
                '"><input type="text" name = "totalgst" id="gstrate ' + counter +
                '" style = "width:115%; height:20%"></td>';
            cols += '<td class="pt-3-half cost_center10"' + counter +
                '" ><select style="word-wrap:break-word;width:100%;" class="costcenter4" id="cost_center' + counter +
                '" name = "cost_center3">{% for element in cost_center_info %}<option value="{{element.cost_center_name}}">{{element.cost_center_name}}</option>{% endfor %}</td>';
            cols += '<td class="pt-3-half" ' + counter +
                '"><input type="text" name = "hsn" id="hsn' + counter +
                '" style = "width:115%; height:20%"></td>';
            cols += '<td class="pt-3-half" ' + counter +
                '"><input type="text" name = "itemdescription" id="itemdescription' + counter +
                '" class = "item_service_ddescription" ' + counter + ' style = "width:105%; height:20%"></td>';
            cols += '<td class="pt-3-half" ' + counter +
                '"><select style="word-wrap:break-word;width:100%;" class="qty4" id="uqc' + counter +
                '" name = "uqc_type">{% for element in uqc_type %}<option value="{{element.uqc}}">{{element.uqc}}</option>{% endfor %}</td>';

            cols += '<td class="pt-3-half" ' + counter +
                '"><input type="text" name = "qty" style = "width:115%; height:20%"></td>';


            cols += '<td class="pt-3-half" ' + counter +
                ' "> <select style="word-wrap:break-word;width:100%;" class="goods4" name = "goods_or_services2">{% for element in goods_or_services %}<option value="{{element.goods_or_services}}">{{element.goods_or_services}}</option>{% endfor %}</td>';
            "{% endif %}"
            "{% if vouchertype == '1' %}"
            cols += '<td class="pt-3-half" ' + counter +
                '"><input type="date" class="startDate" id="startdate' + counter +
                '" name = "startdate" style = "width:110%; height:20%" readonly> </td>';
            cols += '<td class="pt-3-half" ' + counter +
                '"><input type="date" class="endDate" id="enddate' + counter +
                '" name = "enddate" style = "width:110%; height:20%" readonly></td>';
            "{% endif %}"

            "{% if vouchertype == '3' or vouchertype == '4' or vouchertype == '5' or vouchertype == '6' %}"
            cols += '<td class="pt-3-half"  ' + counter +
                '" > <select  style="word-wrap:break-word;width:100%;" class="ref4" id="reference' + counter +
                '">{% for element in reference %}<option value="{{element.ref_type}}" >{{element.ref_type}}</option>{% endfor %}</select></td>';
            cols += '<td class="pt-3-half" ' + counter +
                '"><input type = "text"  style = "width:100%" name = "amount" id = "amount' + counter +
                '" > </td>';
            cols += '<td class="pt-3-half"   ' + counter +
                ' "> <input type = "text" id = "old_reference' + counter +
                '" name = "old_reference" style = "width:100%"  readonly> </td>';
            "{% endif %}"
            var ledger_name = $('#vendor_name').val();

            $.ajax({
                url: '../../billbybill/' + "{{company_code}}",
                data: {
                    ledger_name: ledger_name,
                    csrfmiddlewaretoken: '{{ csrf_token }}',
                },
                dataType: 'json',
                type: 'POST',
                success: function (data) {
                    //alert(data);
                    console.log(data); //This is pure text.
                    console.log(data.result);
                    // condole.log(data.result)
                    var billbybill = data.result;
                    if (billbybill[0][0] == "Yes") {
                        $("#reference_type1").attr("hidden", false);
                        $("#reference_type11").attr("hidden", false);
                        $("#reference_type2").attr("hidden", false);
                    }
                    else {
                        $("#reference_type1").attr("hidden", true)
                        $("#reference_type11").attr("hidden", true)
                        $("#reference_type2").attr("hidden", true)
                    }
                    if (billbybill[0][1] == "Nominal") {
                        $(".cost_center10").attr("hidden", false);
                        console.log("if")
                    }
                    else {
                        console.log("else")
                        $(".cost_center10").attr("hidden", true);

                    }
                    $("#gstnoofshipto").val(billbybill[0][2])
                }
            });

            newRow.append(cols);

            $("#myTable").append(newRow);

            counter++;
            $(".costcenter4").select2();
            $(".qty4").select2();
            $(".goods4").select2();
            $(".ref4").select2();
            $(".expense_items").select2();
            console.log("end");
            "{% if vouchertype == '3' or vouchertype == '4' or vouchertype == '5' or vouchertype == '6' %}"
            $("#reference" + (counter - 1)).on("change", function () {
                console.log("hello")
                if ($("#reference" + (counter - 1)).val() == "Agst Ref") {
                    console.log("You chose agst ref");
                    $("#old_reference" + (counter - 1)).attr("readonly", false);
                } else {
                    $("#old_reference" + (counter - 1)).attr("readonly", true);

                }
            });
            "{% endif %}"



            $(document).ready(function () {

                console.log(counter - 1)

                $("#goods_or_services" + (counter - 1)).on("change", function () {
                    console.log("chahal")

                    if ($("#goods_or_services" + (counter - 1)).val() == "G") {
                        $("#startdate" + (counter - 1)).attr('readonly', true)
                        $("#enddate" + (counter - 1)).attr('readonly', true)
                        $("#qty" + (counter - 1)).attr('readonly', false)
                        $("#uqc_type" + (counter - 1)).attr('readonly', false)
                    } else {
                        $("#startdate" + (counter - 1)).attr('readonly', false)
                        $("#enddate" + (counter - 1)).attr('readonly', false)
                        $("#qty" + (counter - 1)).attr('readonly', true)
                        $("#uqc_type" + (counter - 1)).attr('readonly', true)
                    }
                });


            });

            var date = new Date();
            var day = date.getDate();
            var month = date.getMonth() + 1;
            var year = date.getFullYear();
            if (month < 10) month = "0" + month;
            if (day < 10) day = "0" + day;
            var today = year + "-" + month + "-" + day;
            (/, document.getElementById('startdate', +, (counter, -, 1)).value, =, today;)
            (/, $(".startDate").val(today);)
            (/, document.getElementById('enddate', +, (counter, -, 1)).value, =, today;)
            (/, $(".endDate").val(today))

        });

        $("#myTable").on("click", ".ibtnDel", function (event) {
            $(this).closest("tr").remove();
            counter -= 1
        });

    });

    $(document).ready(function () {

        $("#goods_or_services").on("change", function () {
            console.log("chahal")

            if ($("#goods_or_services").val() == "G") {
                $("#startdate").attr('readonly', true)
                $("#enddate").attr('readonly', true)
                $("#qty").attr('readonly', false)
                $("#uqc_type").attr('readonly', false)
            } else {
                $("#startdate").attr('readonly', false)
                $("#enddate").attr('readonly', false)
                $("#qty").attr('readonly', true)
                $("#uqc_type").attr('readonly', true)
            }
        });


    });
    $("#goods_or_services").on("change", function () {
        console.log("chahal")

        if ($("#goods_or_services").val() == "G") {
            console.log("G")
            $("#startdate").attr('readonly', true)
            $("#enddate").attr('readonly', true)
            $("#qty").attr('readonly', false)
            $("#uqc_type").attr('readonly', false)
        } else {
            console.log("S")
            $("#startdate").attr('readonly', false)
            $("#enddate").attr('readonly', false)
            $("#qty").attr('readonly', true)
            $("#uqc_type").attr('readonly', true)
        }
    });
    $(document).ready(function () {
        $(".vendor_name").select2();
        $(".supply_state").select2();
        $("#expense_head").select2();
        $("#expense_head2").select2();
        $("#expense_hea4").select2();
        $("#expense_head1").select2();
        $("#expense_head3").select2();
        $(".expense55").select2();
        $(".expense21").select2();
        $(".expense22").select2();
        $(".expense23").select2();
        (/, $(".expense57").select2();)
        $(".expense58").select2();
        $(".expense59").select2();

        var counter = 0;

        (/, var, list, =, [];)
        (/, var, goods, =, '{{, goods_or_services, }}';)
        (/, //, list.push(goods);)
        (/, console.log(goods);)
        $("#addrow3").on("click", function () {
            console.log("Hello");
            console.log(counter);
            (/, $('#myTable, tbody').append($('#myTable, tbody, tr:last').clone());)
            var newRow = $("<tr>");
            var cols = "";
            cols +=
                '<td><input type="button" class="ibtnDel btn btn-md btn-sm btn-danger headcol" style=height:2.5em; value="Delete" bgcolor="#E6E6FA"></td>';
            "{% if vouchertype == '1' or vouchertype == '2' %}"
            cols += '<td class="pt-3-half" ' + counter +
                '"><select style="word-wrap:break-word;width:100%;" class="expense52" id="expense_head' + counter +
                '" name = "expense_type2">{% for element in ledgers %}<option value="{{element.ledger_name}}">{{element.ledger_name}}</option>{% endfor %}</td>';
            cols += '<td class="pt-3-half" ' + counter +
                '"><input type="text" name = "taxablevalue2" style = "width:105%; height:20%"> </td>';
            cols += '<td class="pt-3-half" ' + counter +
                '"><input type="text" name = "gst_name2" id="gst' + counter +
                '" style = "width:115%; height:20%"></td>';
            cols += '<td class="pt-3-half" ' + counter +
                '"><input type="text" name = "totalgst2" id="gstrate ' + counter +
                '" style = "width:115%; height:20%"></td>';
            cols += '<td class="pt-3-half cost_center10" ' + counter +
                '" ><select style="word-wrap:break-word;width:100%;" class="costcenter5" id="cost_center' + counter +
                '" name = "cost_center2">{% for element in cost_center_info %}<option value="{{element.cost_center_name}}">{{element.cost_center_name}}</option>{% endfor %}</td>';
            cols += '<td class="pt-3-half" ' + counter +
                '"><input type="text" name = "hsn1" id="hsn' + counter +
                '" style = "width:115%; height:20%"></td>';
            cols += '<td class="pt-3-half" ' + counter +
                '"><input type="text" name = "itemdescription1" id="itemdescription' + counter +
                '" class = "item_service_ddescription" ' + counter + ' style = "width:105%; height:20%"></td>';

            cols += '<td class="pt-3-half" ' + counter +
                '"><input type="text" name = "qty1" style = "width:115%; height:20%"></td>';

            cols += '<td class="pt-3-half" ' + counter +
                '"><select style="word-wrap:break-word;width:100%;" class="qty5" id="uqc' + counter +
                '" name = "uqc_type1">{% for element in uqc_type %}<option value="{{element.uqc}}">{{element.uqc}}</option>{% endfor %}</td>';



            cols += '<td class="pt-3-half" ' + counter +
                ' "> <select style="word-wrap:break-word;width:100%;" class="goods5" name = "goods_or_services1">{% for element in goods_or_services %}<option value="{{element.goods_or_services}}">{{element.goods_or_services}}</option>{% endfor %}</td>';
            "{% endif %}"
            "{% if vouchertype == '1' %}"
            cols += '<td class="pt-3-half" ' + counter +
                '"><input type="date" class="startDate" id="startdate' + counter +
                '" name = "startdate1" style = "width:110%; height:20%" readonly> </td>';
            cols += '<td class="pt-3-half" ' + counter +
                '"><input type="date" class="endDate" id="enddate' + counter +
                '" name = "enddate1" style = "width:110%; height:20%" readonly></td>';
            "{% endif %}"

            "{% if vouchertype == '3' or vouchertype == '4' or vouchertype == '5' or vouchertype == '6' %}"
            cols += '<td class="pt-3-half"  ' + counter +
                '" > <select  style="word-wrap:break-word;width:100%;" class="ref5" id="reference' + counter +
                '">{% for element in reference %}<option value="{{element.ref_type}}" >{{element.ref_type}}</option>{% endfor %}</select></td>';
            cols += '<td class="pt-3-half" ' + counter +
                '"><input type = "text"  style = "width:100%" name = "amount" id = "amount' + counter +
                '" > </td>';
            cols += '<td class="pt-3-half"   ' + counter +
                ' "> <input type = "text" id = "old_reference' + counter +
                '" name = "old_reference" style = "width:100%"  readonly> </td>';
            "{% endif %}"
            var ledger_name = $('#vendor_name').val();

            $.ajax({
                url: '../../billbybill/' + "{{company_code}}",
                data: {
                    ledger_name: ledger_name,
                    csrfmiddlewaretoken: '{{ csrf_token }}',
                },
                dataType: 'json',
                type: 'POST',
                success: function (data) {
                    //alert(data);
                    console.log(data); //This is pure text.
                    console.log(data.result);
                    // condole.log(data.result)
                    var billbybill = data.result;
                    if (billbybill[0][0] == "Yes") {
                        $("#reference_type1").attr("hidden", false);
                        $("#reference_type11").attr("hidden", false);
                        $("#reference_type2").attr("hidden", false);
                    }
                    else {
                        $("#reference_type1").attr("hidden", true)
                        $("#reference_type11").attr("hidden", true)
                        $("#reference_type2").attr("hidden", true)
                    }
                    if (billbybill[0][1] == "Nominal") {
                        $(".cost_center10").attr("hidden", false);
                        console.log("if")
                    }
                    else {
                        console.log("else")
                        $(".cost_center10").attr("hidden", true);

                    }
                    $("#gstnoofshipto").val(billbybill[0][2])
                }
            });


            console.log(counter)

            newRow.append(cols);
            $("#myTable3").append(newRow);
            counter++;
            $(".costcenter5").select2();
            $(".qty5").select2();
            $(".goods5").select2();
            $(".ref5").select2();
            $(".expense52").select2();
            "{% if vouchertype == '3' or vouchertype == '4' or vouchertype == '5' or vouchertype == '6' %}"
            $("#reference" + (counter - 1)).on("change", function () {
                console.log("hello")
                if ($("#reference" + (counter - 1)).val() == "Agst Ref") {
                    console.log("You chose agst ref");
                    $("#old_reference" + (counter - 1)).attr("readonly", false);
                } else {
                    $("#old_reference" + (counter - 1)).attr("readonly", true);

                }
            });
            "{% endif %}"



            $(document).ready(function () {

                console.log(counter - 1)

                $("#goods_or_services" + (counter - 1)).on("change", function () {
                    console.log("chahal")

                    if ($("#goods_or_services" + (counter - 1)).val() == "G") {
                        $("#startdate" + (counter - 1)).attr('readonly', true)
                        $("#enddate" + (counter - 1)).attr('readonly', true)
                        $("#qty" + (counter - 1)).attr('readonly', false)
                        $("#uqc_type" + (counter - 1)).attr('readonly', false)
                    } else {
                        $("#startdate" + (counter - 1)).attr('readonly', false)
                        $("#enddate" + (counter - 1)).attr('readonly', false)
                        $("#qty" + (counter - 1)).attr('readonly', true)
                        $("#uqc_type" + (counter - 1)).attr('readonly', true)
                    }
                });


            });

            var date = new Date();
            var day = date.getDate();
            var month = date.getMonth() + 1;
            var year = date.getFullYear();
            if (month < 10) month = "0" + month;
            if (day < 10) day = "0" + day;
            var today = year + "-" + month + "-" + day;
            (/, document.getElementById('startdate', +, (counter, -, 1)).value, =, today;)
            (/, $(".startDate").val(today);)
            (/, document.getElementById('enddate', +, (counter, -, 1)).value, =, today;)
            (/, $(".endDate").val(today))




        });
        $("#myTable3").on("click", ".ibtnDel", function (event) {
            $(this).closest("tr").remove();
            counter -= 1
        });
    });
    $(document).ready(function () {
        $("#goods_or_services").on("change", function () {
            console.log("chahal")

            if ($("#goods_or_services").val() == "G") {
                $("#startdate").attr('readonly', true)
                $("#enddate").attr('readonly', true)
                $("#qty").attr('readonly', false)
                $("#uqc_type").attr('readonly', false)
            } else {
                $("#startdate").attr('readonly', false)
                $("#enddate").attr('readonly', false)
                $("#qty").attr('readonly', true)
                $("#uqc_type").attr('readonly', true)
            }
        });


    });
    $("#goods_or_services").on("change", function () {
        console.log("chahal")

        if ($("#goods_or_services").val() == "G") {
            console.log("G")
            $("#startdate").attr('readonly', true)
            $("#enddate").attr('readonly', true)
            $("#qty").attr('readonly', false)
            $("#uqc_type").attr('readonly', false)
        } else {
            console.log("S")
            $("#startdate").attr('readonly', false)
            $("#enddate").attr('readonly', false)
            $("#qty").attr('readonly', true)
            $("#uqc_type").attr('readonly', true)
        }
    });

    $(document).ready(function () {
        var counter = 0;
        console.log("counter")


        $("#addrow1").on("click", function () {
            console.log("addrow1");

            var newRow = $("<tr>");
            var cols = "";
            cols +=
                '<td><input type="button" class="ibtnDel btn btn-md btn-sm btn-danger headcol" style=height:2.5em; value="Delete" bgcolor="#E6E6FA"></td>';
            "{% if vouchertype == '1' or vouchertype == '2' %}"
            cols += '<td class="pt-3-half" ' + counter +
                '"><select style="word-wrap:break-word;width:100%;" class="expense21" name = "expense_type">{% for element in ledgers %}<option value="{{element.ledger_name}}">{{element.ledger_name}}</option>{% endfor %}</td>';
            cols += '<td class="pt-3-half" ' + counter +
                '"><input type="text" name = "taxablevalue3" style = "width:105%; height:20%"> </td>';
            cols += '<td class="pt-3-half" ' + counter +
                '"><input type="text" name = "gst_name4" style = "width:105%; height:20%"> </td>';
            cols += '<td class="pt-3-half" ' + counter +
                '"><input type="text" name = "amount" id="gst' + counter +
                '" style = "width:115%; height:20%"></td>';
            cols += '<td class="pt-3-half cost_center10" ' + counter +
                '" ><select style="word-wrap:break-word;width:100%;" class="costcenter6" name = "cost_center8">{% for element in cost_center_info %}<option value="{{element.cost_center_name}}">{{element.cost_center_name}}</option>{% endfor %}</td>';
            console.log(counter)
            "{% endif %}"
            "{% if vouchertype == '3' or vouchertype == '4' or vouchertype == '5' or vouchertype == '6' %}"
            cols += '<td class="pt-3-half" ' + counter +
                '"><select style="word-wrap:break-word;width:100%;" class="expense25" name = "expense_type">{% for element in ledgers %}<option value="{{element.ledger_name}}">{{element.ledger_name}}</option>{% endfor %}</td>';
            cols += '<td class="pt-3-half" ' + counter +
                '"><input type = "text"  style = "width:100%" name = "amount" id = "amount' +
                counter + '" > </td>';
            cols += '<td class="pt-3-half"  ' + counter +
                '" > <select  style="word-wrap:break-word;width:100%;" class="costcenter6" id="reference' +
                counter +
                '">{% for element in reference %}<option value="{{element.ref_type}}" >{{element.ref_type}}</option>{% endfor %}</select></td>';

            cols += '<td class="pt-3-half"   ' + counter +
                ' "> <input type = "text" id = "old_reference' + counter +
                '" name = "old_reference" style = "width:100%"  readonly> </td>';
            cols += '<td class="pt-3-half cost_center10" ' + counter +
                '" ><select style="word-wrap:break-word;width:100%;" name = "cost_center8">{% for element in cost_center_info %}<option value="{{element.cost_center_name}}">{{element.cost_center_name}}</option>{% endfor %}</td>';

            "{% endif %}"
            var ledger_name = $('#vendor_name').val();

            $.ajax({
                url: '../../billbybill/' + "{{company_code}}",
                data: {
                    ledger_name: ledger_name,
                    csrfmiddlewaretoken: '{{ csrf_token }}',
                },
                dataType: 'json',
                type: 'POST',
                success: function (data) {
                    //alert(data);
                    console.log(data); //This is pure text.
                    console.log(data.result);
                    // condole.log(data.result)
                    var billbybill = data.result;
                    if (billbybill[0][0] == "Yes") {
                        $("#reference_type1").attr("hidden", false);
                        $("#reference_type11").attr("hidden", false);
                        $("#reference_type2").attr("hidden", false);
                    }
                    else {
                        $("#reference_type1").attr("hidden", true)
                        $("#reference_type11").attr("hidden", true)
                        $("#reference_type2").attr("hidden", true)
                    }
                    if (billbybill[0][1] == "Nominal") {
                        $(".cost_center10").attr("hidden", false);
                        console.log("if")
                    }
                    else {
                        console.log("else")
                        $(".cost_center10").attr("hidden", true);

                    }
                    $("#gstnoofshipto").val(billbybill[0][2])
                }
            });
            newRow.append(cols);
            $("#myTable1").append(newRow);
            counter++;
            $(".costcenter6").select2();

            $(".ref6").select2();
            $(".expense21").select2();
            $(".expense25").select2();


        });

        $("#myTable1").on("click", ".ibtnDel", function (event) {
            $(this).closest("tr").remove();
            counter -= 1




        });



        $(document).ready(function () {

            console.log(counter - 1)

            $("#goods_or_services" + (counter - 1)).on("change", function () {
                console.log("chahal")

                if ($("#goods_or_services" + (counter - 1)).val() == "G") {
                    $("#startdate" + (counter - 1)).attr('readonly', true)
                    $("#enddate" + (counter - 1)).attr('readonly', true)
                    $("#qty" + (counter - 1)).attr('readonly', false)
                    $("#uqc_type" + (counter - 1)).attr('readonly', false)
                } else {
                    $("#startdate" + (counter - 1)).attr('readonly', false)
                    $("#enddate" + (counter - 1)).attr('readonly', false)
                    $("#qty" + (counter - 1)).attr('readonly', true)
                    $("#uqc_type" + (counter - 1)).attr('readonly', true)
                }
            });


        });

        var date = new Date();
        var day = date.getDate();
        var month = date.getMonth() + 1;
        var year = date.getFullYear();
        if (month < 10) month = "0" + month;
        if (day < 10) day = "0" + day;
        var today = year + "-" + month + "-" + day;
        (/, document.getElementById('startdate', +, (counter, -, 1)).value, =, today;)
        $(".startDate").val(today);
        (/, document.getElementById('enddate', +, (counter, -, 1)).value, =, today;)
        $(".endDate").val(today)





        $("#myTable").on("click", ".ibtnDel", function (event) {
            $(this).closest("tr").remove();
            counter -= 1
        });
    });


    $(document).ready(function () {
        var counter = 0;

        $("#addrow2").on("click", function () {
            var newRow = $("<tr>");
            var cols = "";
            cols +=
                '<td><input type="button" class="ibtnDel btn btn-md btn-sm btn-danger headcol" style=height:2.5em; value="Delete" bgcolor="#E6E6FA"></td>';
            "{% if vouchertype == '1' or vouchertype == '2' %}"
            cols += '<td class="pt-3-half" ' + counter +
                '"><select style="word-wrap:break-word;width:100%;" class="expense42" name = "expense_type1">{% for element in ledgers %}<option value="{{element.ledger_name}}">{{element.ledger_name}}</option>{% endfor %}</td>';
            cols += '<td class="pt-3-half" ' + counter +
                '"><input type="text" name = "taxablevalue5" style = "width:105%; height:20%"> </td>';
            cols += '<td class="pt-3-half" ' + counter +
                '"><input type="text" name = "gst_name1" style = "width:105%; height:20%" > </td>';
            cols += '<td class="pt-3-half" ' + counter +
                '"><input type="text" name = "totalgst1" id="gst' + counter +
                '" style = "width:115%; height:20%"></td>';
            cols += '<td class="pt-3-half cost_center10"' + counter +
                '" ><select style="word-wrap:break-word;width:100%;" class="costcenter7" name = "cost_center1">{% for element in cost_center_info %}<option value="{{element.cost_center_name}}">{{element.cost_center_name}}</option>{% endfor %}</td>';
            console.log(counter)
            "{% endif %}"
            "{% if vouchertype == '3' or vouchertype == '4' or vouchertype == '5' or vouchertype == '6' %}"
            cols += '<td class="pt-3-half"  ' + counter +
                '" > <select  style="word-wrap:break-word;width:100%;" class="ref7" id="reference' +
                counter +
                '">{% for element in reference %}<option value="{{element.ref_type}}" >{{element.ref_type}}</option>{% endfor %}</select></td>';
            cols += '<td class="pt-3-half" ' + counter +
                '"><input type = "text"  style = "width:100%" name = "amount" id = "amount' +
                counter + '" > </td>';
            cols += '<td class="pt-3-half"   ' + counter +
                ' "> <input type = "text" id = "old_reference' + counter +
                '" name = "old_reference" style = "width:100%"  readonly> </td>';
            "{% endif %}"
            var ledger_name = $('#vendor_name').val();

            $.ajax({
                url: '../../billbybill/' + "{{company_code}}",
                data: {
                    ledger_name: ledger_name,
                    csrfmiddlewaretoken: '{{ csrf_token }}',
                },
                dataType: 'json',
                type: 'POST',
                success: function (data) {
                    //alert(data);
                    console.log(data); //This is pure text.
                    console.log(data.result);
                    // condole.log(data.result)
                    var billbybill = data.result;
                    if (billbybill[0][0] == "Yes") {
                        $("#reference_type1").attr("hidden", false);
                        $("#reference_type11").attr("hidden", false);
                        $("#reference_type2").attr("hidden", false);
                    }
                    else {
                        $("#reference_type1").attr("hidden", true)
                        $("#reference_type11").attr("hidden", true)
                        $("#reference_type2").attr("hidden", true)
                    }
                    if (billbybill[0][1] == "Nominal") {
                        $(".cost_center10").attr("hidden", false);
                        console.log("if")
                    }
                    else {
                        console.log("else")
                        $(".cost_center10").attr("hidden", true);

                    }
                    $("#gstnoofshipto").val(billbybill[0][2])
                }
            });


            newRow.append(cols);
            $("#myTable2").append(newRow);
            counter++;
            $(".costcenter7").select2();
            $(".ref7").select2();
            $(".expense42").select2();



        });

        $("#myTable2").on("click", ".ibtnDel", function (event) {
            $(this).closest("tr").remove();
            counter -= 1



        });



        $(document).ready(function () {

            console.log(counter - 1)

            $("#goods_or_services1" + (counter - 1)).on("change", function () {
                console.log("chahal")

                if ($("#goods_or_services1" + (counter - 1)).val() == "G") {
                    $("#startdate1" + (counter - 1)).attr('readonly', true)
                    $("#enddate1" + (counter - 1)).attr('readonly', true)
                    $("#qty1" + (counter - 1)).attr('readonly', false)
                    $("#uqc_type1" + (counter - 1)).attr('readonly', false)
                } else {
                    $("#startdate1" + (counter - 1)).attr('readonly', false)
                    $("#enddate1" + (counter - 1)).attr('readonly', false)
                    $("#qty1" + (counter - 1)).attr('readonly', true)
                    $("#uqc_type1" + (counter - 1)).attr('readonly', true)
                }
            });


        });

        var date = new Date();
        var day = date.getDate();
        var month = date.getMonth() + 1;
        var year = date.getFullYear();
        if (month < 10) month = "0" + month;
        if (day < 10) day = "0" + day;
        var today = year + "-" + month + "-" + day;

        $(".startDate1").val(today);

        $(".endDate1").val(today)





        $("#myTable2").on("click", ".ibtnDel", function (event) {
            $(this).closest("tr").remove();
            counter -= 1
        });
    });

    function next_pdf(num_of_pages, basename) {

        if (i < num_of_pages - 1) {
            i++;
            $("#value_x").attr("value", i);
            (/, $("#a_load_pdf").attr("href",, "../loadPDF/", +, i);)
            $("#pdf_file").attr("src", "../media/files/" + basename + i + ".pdf");
        }
        if (i == num_of_pages - 1) {
            $("#next_pdf").attr('readonly', true);
        } else {
            $("#next_pdf").attr('readonly', false);
        }
        if (i == parseFloat('{{x}}')) {
            $("#load_pdf").attr("hidden", true);
        } else {
            $("#load_pdf").attr("hidden", false);
        }
        if (i == 0) {
            $("#prev_pdf").attr('readonly', true);
        } else {
            $("#prev_pdf").attr('readonly', false);
        }
    }

    function prev_pdf(num_of_pages, basename) {
        if (i > 0) {
            i--;
            $("#value_x").attr("value", i);
            (/, $("#a_load_pdf").attr("href",, "../loadPDF/", +, i);)
            $("#pdf_file").attr("src", "../media/files/" + basename + i + ".pdf");
        }
        if (i == num_of_pages - 1) {
            $("#next_pdf").attr('readonly', true);
        } else {
            $("#next_pdf").attr('readonly', false);
        }
        if (i == 0) {
            $("#prev_pdf").attr('readonly', true);
        } else {
            $("#prev_pdf").attr('readonly', false);
        }
        if (i == parseFloat('{{x}}')) {
            $("#load_pdf").attr("hidden", true);
        } else {
            $("#load_pdf").attr("hidden", false);
        }
    }

    $('#whether_invoice_in_inr').click(function () {
        if ($("#whether_invoice_in_inr").prop("checked") == false) {
            $("#exchange_rate").attr("hidden", false);
            $("#foreign_currency_div").attr("hidden", false);
        }
        if ($("#whether_invoice_in_inr").prop("checked") == true) {
            $("#exchange_rate").attr("hidden", true);
            $("#foreign_currency_div").attr("hidden", true);
        }
    });

    $('#whether_invoice_in_inr1').click(function () {
        if ($("#whether_invoice_in_inr1").prop("checked") == false) {
            $("#exchange_rate1").attr("hidden", false);
            $("#foreign_currency_div1").attr("hidden", false);
        }
        if ($("#whether_invoice_in_inr1").prop("checked") == true) {
            $("#exchange_rate1").attr("hidden", true);
            $("#foreign_currency_div1").attr("hidden", true);
        }
    });

    $('#supply_state').on("change", function () {
        if ($(this).val() == "Other Countries") {
            $("#supply_countryy").attr("hidden", false);
        }
        if ($(this).val() != "Other Countries") {
            $("#supply_countryy").attr("hidden", true);
        }
    });
    $('#supply_state1').on("change", function () {
        if ($(this).val() == "Other Countries") {
            $("#supply_countryy1").attr("hidden", false);
        }
        if ($(this).val() != "Other Countries") {
            $("#supply_countryy1").attr("hidden", true);
        }
    });

    $('#taxable_value').on("change", function () {

        var igst = parseFloat($('#igst').val())
        if (isNaN(igst))
            var igst1 = 0;
        else {
            var igst1 = igst
        }
        var cgst = parseFloat($('#cgst').val())

        if (isNaN(cgst)) {
            var cgst1 = 0;
        } else {
            var cgst1 = cgst
        }

        var sgst = parseFloat($('#sgst').val())

        if (isNaN(sgst))
            var sgst1 = 0;
        else {
            var sgst1 = sgst
        }
        console.log(parseFloat($('#taxable_value').val()))
        console.log(igst1)
        console.log(cgst1)
        console.log(sgst1)
        var sum = parseFloat($('#taxable_value').val()) + igst1 + sgst1 + cgst1;
        var sumtax = parseFloat($('#taxable_value').val())

        $('#invoice_value').val(sum);
        $('#amount').val(sumtax);
        $('#gstrate').val(Math.round((igst1 + sgst1 + cgst1) / sumtax * 100))
        $('#totalgst').val(sumtax + (igst1 + sgst1 + cgst1) / sumtax * 100 * 100)
        $('#amount1').val(sumtax);
        $('#gstrate1').val(Math.round((igst1 + sgst1 + cgst1) / sumtax * 100))
        $('#totalgst1').val(sumtax + (igst1 + sgst1 + cgst1) / sumtax * 100)

    });

    $("#discount3").on("change", function () {
        var dis = $("#discount3").val()
        $("#discount3").val(dis);
    });

    $('#discount3').on("change", function () {
        var bcd = parseFloat($('#discount3').val())
        var taxable = parseFloat($('#taxable_value').val())
        if (isNaN(taxable))
            var taxable1 = 0;
        else {
            var taxable1 = taxable;
        }
        if (isNaN(bcd))
            var bcd1 = 0;
        else {
            var bcd1 = bcd;
        }
        var sumtax = taxable1 - bcd1
        var igst = parseFloat($('#igst').val())
        if (isNaN(igst))
            var igst1 = 0;
        else {
            var igst1 = igst
        }
        var cgst = parseFloat($('#cgst').val())

        if (isNaN(cgst)) {
            var cgst1 = 0;
        } else {
            var cgst1 = cgst
        }

        var sgst = parseFloat($('#sgst').val())

        if (isNaN(sgst))
            var sgst1 = 0;
        else {
            var sgst1 = sgst
        }
        console.log(parseFloat($('#taxable_value').val()))
        console.log(igst1)
        console.log(cgst1)
        console.log(sgst1)
        var sum = sumtax + igst1 + sgst1 + cgst1;
        (/, var, sumtax, =, parseFloat($('#taxable_value').val()))

        $('#amount').val(sumtax);
        $('#amount1').val(sumtax);
        $('#totalgst').val(sumtax + (igst1 + sgst1 + cgst1))
        $('#totalgst1').val(sumtax + (igst1 + sgst1 + cgst1))
    });

    $('#igst').on("change", function () {
        var bcd = parseFloat($('#discount3').val())
        var taxable = parseFloat($('#taxable_value').val())
        if (isNaN(taxable))
            var taxable1 = 0;
        else {
            var taxable1 = taxable;
        }
        if (isNaN(bcd))
            var bcd1 = 0;
        else {
            var bcd1 = bcd;
        }
        var sumtax = taxable1 - bcd1
        console.log("drswewaresresrfew")
        var igst = parseFloat($('#igst').val())
        if (isNaN(igst))
            var igst1 = 0;
        else {
            var igst1 = igst
        }
        var cgst = parseFloat($('#cgst').val())

        if (isNaN(cgst)) {
            var cgst1 = 0;
        } else {
            var cgst1 = cgst
        }

        var sgst = parseFloat($('#sgst').val())

        if (isNaN(sgst))
            var sgst1 = 0;
        else {
            var sgst1 = sgst
        }
        console.log(parseFloat($('#taxable_value').val()))
        console.log(igst1)
        console.log(cgst1)
        console.log(sgst1)
        var sum = sumtax + igst1 + sgst1 + cgst1;
        (/, var, sumtax, =, parseFloat($('#taxable_value').val()))


        $('#invoice_value').val(sum);
        (/, $('#amount').val(sumtax);)
        var sumgst = igst1 + cgst1 + sgst1;
        (/, $('#totalgst').val(sumgst);)
        $('#gstrate').val(Math.round((igst1 + sgst1 + cgst1) / sumtax * 100))
        $('#totalgst').val(sumtax + (igst1 + sgst1 + cgst1))
        (/, $('#amount1').val(sumtax);)
        $('#gstrate1').val(Math.round((igst1 + sgst1 + cgst1) / sumtax * 100))
        $('#totalgst1').val(sumtax + (igst1 + sgst1 + cgst1))
    });
    $('#cgst').on("change", function () {
        var bcd = parseFloat($('#discount3').val())
        var taxable = parseFloat($('#taxable_value').val())
        if (isNaN(taxable))
            var taxable1 = 0;
        else {
            var taxable1 = taxable;
        }
        if (isNaN(bcd))
            var bcd1 = 0;
        else {
            var bcd1 = bcd;
        }
        var sumtax = taxable1 - bcd1
        console.log("drswewaresresrfew")
        var igst = parseFloat($('#igst').val())
        if (isNaN(igst))
            var igst1 = 0;
        else {
            var igst1 = igst
        }
        var cgst = parseFloat($('#cgst').val())

        if (isNaN(cgst)) {
            var cgst1 = 0;
        } else {
            var cgst1 = cgst
        }

        var sgst = parseFloat($('#sgst').val())

        if (isNaN(sgst))
            var sgst1 = 0;
        else {
            var sgst1 = sgst
        }
        console.log(parseFloat($('#taxable_value').val()))
        console.log(igst1)
        console.log(cgst1)
        console.log(sgst1)
        var sum = sumtax + igst1 + sgst1 + cgst1;
        (/, var, sumtax, =, parseFloat($('#taxable_value').val()))

        $('#invoice_value').val(sum);
        (/, $('#amount').val(sumtax);)
        var sumgst = igst1 + cgst1 + sgst1;
        (/, $('#totalgst').val(sumgst);)
        $('#gstrate').val(Math.round((igst1 + sgst1 + cgst1) / sumtax * 100))
        $('#totalgst').val(sumtax + (igst1 + sgst1 + cgst1))
        (/, $('#amount1').val(sumtax);)
        $('#gstrate1').val(Math.round((igst1 + sgst1 + cgst1) / sumtax * 100))
        $('#totalgst1').val(sumtax + (igst1 + sgst1 + cgst1))
    });
    $('#sgst').on("change", function () {
        var bcd = parseFloat($('#discount3').val())
        var taxable = parseFloat($('#taxable_value').val())
        if (isNaN(taxable))
            var taxable1 = 0;
        else {
            var taxable1 = taxable;
        }
        if (isNaN(bcd))
            var bcd1 = 0;
        else {
            var bcd1 = bcd;
        }
        var sumtax = taxable1 - bcd1
        console.log("drswewaresresrfew")
        var igst = parseFloat($('#igst').val())
        if (isNaN(igst))
            var igst1 = 0;
        else {
            var igst1 = igst
        }
        var cgst = parseFloat($('#cgst').val())

        if (isNaN(cgst)) {
            var cgst1 = 0;
        } else {
            var cgst1 = cgst
        }

        var sgst = parseFloat($('#sgst').val())

        if (isNaN(sgst))
            var sgst1 = 0;
        else {
            var sgst1 = sgst
        }
        console.log(parseFloat($('#taxable_value').val()))
        console.log(igst1)
        console.log(cgst1)
        console.log(sgst1)
        var sum = sumtax + igst1 + sgst1 + cgst1;
        (/, var, sumtax, =, parseFloat($('#taxable_value').val()))

        $('#invoice_value').val(sum);
        (/, $('#amount').val(sumtax);)
        var sumgst = igst1 + cgst1 + sgst1;
        (/, $('#totalgst').val(sumgst);)
        $('#gstrate').val(Math.round((igst1 + sgst1 + cgst1) / sumtax * 100))
        $('#totalgst').val(sumtax + (igst1 + sgst1 + cgst1))
        (/, $('#amount1').val(sumtax);)
        $('#gstrate1').val(Math.round((igst1 + sgst1 + cgst1) / sumtax * 100))
        $('#totalgst1').val(sumtax + (igst1 + sgst1 + cgst1))
    });

    $('#sgst').on("change", function () {

        var igst = parseFloat($('#igst').val())
        var sgst = parseFloat($('#sgst').val())
        var cgst = parseFloat($('#cgst').val())
        if (isNaN(igst) && isNaN(sgst) && isNaN(cgst)) {
            $("#rcm_applicable").prop("checked", true)
        } else {
            $("#rcm_applicable").prop("checked", false)
        }
    });

    $('#cgst').on("change", function () {

        var igst = parseFloat($('#igst').val())
        var sgst = parseFloat($('#sgst').val())
        var cgst = parseFloat($('#cgst').val())
        if (isNaN(igst) && isNaN(sgst) && isNaN(cgst)) {
            $("#rcm_applicable").prop("checked", true)
        } else {
            $("#rcm_applicable").prop("checked", false)
        }
    });


    $('#igst').on("change", function () {

        var igst = parseFloat($('#igst').val())
        console.log(igst)
        var sgst = parseFloat($('#sgst').val())
        console.log(sgst)
        var cgst = parseFloat($('#cgst').val())
        console.log(cgst)
        if (isNaN(igst) & isNaN(sgst) & isNaN(cgst)) {
            console.log("vgdfsgdvhs")
            $("#rcm_applicable").prop("checked", true)
        } else {
            console.log("RCM")
            $("#rcm_applicable").prop("checked", false)
        }
    });

    $('#whether_import').click(function () {
        if ($("#whether_import").prop("checked") == false) {
            $("#shipping_bill_no").attr("hidden", true);
            $("#shipping_bill_date_date").attr("hidden", true);
        }
        if ($("#whether_import").prop("checked") == true) {

            $("#shipping_bill_no").attr("hidden", false);
            $("#shipping_bill_date_date").attr("hidden", false);
        }
    });

    $('#cess').click(function () {
        if ($("#cess").prop("checked") == false) {
            $("#amount_of_tax_cess").attr("hidden", true);
            $("#amount_of_itc_cess").attr("hidden", true);
        }
        if ($("#cess").prop("checked") == true) {
            $("#amount_of_tax_cess").attr("hidden", false);
            $("#amount_of_itc_cess").attr("hidden", false);
        }
    });
    function ok() {
        $("#modal").modal('toggle')

    }

    function editaccounting(company_code) {

        console.log('dfhsdjkfh');
        var ledger_name = $('#vendor_name').val();
        var invoice_date = $('#theDate').val();

        $.ajax({
            url: '../../editaccounting/' + company_code,
            data: {
                ledger_name: ledger_name,
                invoice_date: invoice_date,
                csrfmiddlewaretoken: '{{ csrf_token }}',
            },
            dataType: 'json',
            type: 'POST',
            success: function (data) {
                //alert(data);
                console.log(data); //This is pure text.
                console.log(data.result);
                $("#type_of_service").attr("value", data.result);
                $("#type_of_sale").attr("value", data.type_of_sale);
                $("#whether_capital_or_input").attr("value", data.capital_or_input1);
                $("#tds_section").attr("value", data.tdssectioninfo);
                $("#tds_percentage").attr("value", data.tdspercentageinfo);

            }
        });
    }
    $("#reference").on("change", function () {
        if ($("#reference").val() == "Agst Ref") {
            console.log("You chose anst ref");
            $("#tds_applicable").attr("readonly", false);
            (/, $("#tds_applicable_th").attr("readonly",, false);)
            $("#tds_applicable_2").attr("readonly", false);
            $("#old_reference").attr("readonly", false);
            (/, $("#old_reference_th").attr("readonly",, false);)
        } else {
            $("#tds_applicable").attr("readonly", true);
            $("#tds_amount").attr("readonly", true);
            $("#tds_amount_override").attr("readonly", true);
            (/, $("#tds_applicable_th").attr("readonly",, true);)
            (/, $("#tds_amount_th").attr("readonly",, true);)
            (/, $("#tds_amount_override_th").attr("readonly",, true);)
            $("#tds_applicable_2").attr("readonly", true);
            $("#old_reference").attr("readonly", true);
            (/, $("#old_reference_th").attr("readonly",, true);)
        }
    });
    $(document).ready(function () {
        var date = new Date();
        var day = date.getDate();
        var month = date.getMonth() + 1;
        var year = date.getFullYear();
        if (month < 10) month = "0" + month;
        if (day < 10) day = "0" + day;
        var today = year + "-" + month + "-" + day;
        (/, document.getElementById('theDate').value, =, today;)
        document.getElementById('voucher_date').value = today;
        (/, document.getElementById('startdate').value, =, today;)
        (/, document.getElementById('enddate').value, =, today;)
        document.getElementById('theDate').value = today;
        document.getElementById('shipping_bill_date_date1').value = today;
        document.getElementById('theDate1').value = today;
        (/, document.getElementById('shipping_bill_date_date').value, =, today;)

        if (i == parseFloat('{{x}}')) {
            $("#load_pdf").attr("hidden", true);
        } else {
            $("#load_pdf").attr("hidden", false);
        }
        if (i == 0) {
            $("#prev_pdf").attr('readonly', true);
        } else {
            $("#prev_pdf").attr('readonly', false);
        }
        if (i == parseFloat("{{ num_of_pages }}") - 1) {
            $("#next_pdf").attr('readonly', true);
        } else {
            $("#next_pdf").attr('readonly', false);
        }
    });
    $("#goods_or_services").on("change", function () {
        console.log("chahal")

        if ($("#goods_or_services").val() == "G") {
            console.log("G")
            $("#startdate").attr('readonly', true)
            $("#enddate").attr('readonly', true)
            $("#qty").attr('readonly', false)
            $("#uqc_type").attr('readonly', false)
        } else {
            console.log("S")
            $("#startdate").attr('readonly', false)
            $("#enddate").attr('readonly', false)
            $("#qty").attr('readonly', true)
            $("#uqc_type").attr('readonly', true)
        }
    });
</script>


<div class="main-panel">
    <div class="wrapper-editor">
        <div class="row d-flex justify-content-center modalWrapper">
            <div class="modal fade bd-example-modal-xl addNewInputs" id="modalAdd" role="dialog"
                aria-labelledby="modalAdd" aria-hidden="true">
                <div class="modal-dialog modal-xl" role="document" style="max-width: 640px;">
                    <div class="modal-content container">
                        <div class="modal-header text-center">
                            <h4 class="modal-title w-100 font-weight-bold text-primary ml-5">Add new Ledger</h4>
                            <button type="button" class="close text-primary" data-dismiss="modal" aria-label="Close">
                                <span aria-hidden="true">&times;</span>
                            </button>
                        </div>
                        <div class="content">
                            <div class="main-content-inner slideshow-container">
                                <form action="{% url 'ledger1' company_code %}" method="post">
                                    {% csrf_token %}
                                    <input type="text" name="vouchertype" value="{{vouchertype}}" hidden>
                                    <!-- <div class="container mySlides fade"> -->
                                    <div class="row" style="padding-bottom: 1px;">
                                        <div class="form-group col-lg-4">
                                            <div class="form-group">
                                                <label for="example-text-input" class="col-form-label">Ledger Name<span
                                                        style=color:red>*</span></label>
                                                <input class="form-control" type="text" name="ledger_name" value=""
                                                    id="ledger_name1" required>
                                            </div>
                                        </div>
                                        <div class="form-group col-lg-4">
                                            <div class="form-group">
                                                <label for="example-text-input" class="col-form-label">Vendor
                                                    Name</label>
                                                <input class="form-control" type="text" name="vendor_name" value=""
                                                    id="vendor_name1">
                                            </div>
                                        </div>
                                        <div class="form-group col-lg-4">
                                            <div class="form-group">
                                                <label for="example-text-input" class="col-form-label">Alias</label>
                                                <input class="form-control" type="text" name="alias" value=""
                                                    id="alias1">
                                            </div>
                                        </div>
                                    </div>
                                    <div class="row">
                                        <div class="form-group col-lg-4">
                                            <div class="form-group">
                                                <label for="example-text-input" class="col-form-label">Head Name<span
                                                        style=color:red>*</span></label>
                                                <select style="width:160px;" class="custom-select" id="head1"
                                                    name="head" style="width:160px;" required>
                                                    {% for element in head %}
                                                    <option value="{{element.head}}"
                                                        {% if element.head == 'equitity share Capital' %}selected{% endif %}>
                                                        {{element.head}}</option>
                                                    {% endfor %}
                                                </select>
                                            </div>
                                        </div>
                                        <div class="form-group col-lg-4">
                                            <div class="form-group">
                                                <label class="col-form-label">Inward outward<span
                                                        style=color:red>*</span></label>
                                                <select style="width:160px;" class="custom-select" id="inward_outward1"
                                                    name="inward_outward" required>
                                                    <!-- <option value="NA">NA</option> -->
                                                    {% for element in inward_outward %}
                                                    <option value="{{element.inward_outward}}"
                                                        {% if element.inward_outward == 'NA' %}selected{% endif %}>
                                                        {{element.inward_outward}}</option>
                                                    {% endfor %}
                                                </select>
                                            </div>
                                        </div>
                                        <div class="form-group col-lg-4">
                                            <div class="form-group">
                                                <label class="col-form-label">Type of Account</label>
                                                <select style="width:160px;" class="custom-select" id="type_of_account1"
                                                    name="type_of_account">
                                                    <option value="NA">NA</option>
                                                    {% for element in type_of_account %}
                                                    <option value="{{element.type_of_account}}"
                                                        {% if element.type_of_account == 'Real' %}selected{% endif %}>
                                                        {{element.type_of_account}}</option>
                                                    {% endfor %}
                                                </select>
                                            </div>
                                        </div>

                                    </div>
                                    <div class="row">
                                        <div class="form-group col-lg-4">
                                            <div class="form-group" id="type_of_registration1211">
                                                <label class="col-form-label">Type of Registration</label>
                                                <select style="width:160px;" class="custom-select"
                                                    id="type_of_registration1" name="type_of_registration">
                                                    <option value="NA">NA</option>
                                                    {% for element in type_of_registration %}
                                                    <option value="{{element.type_of_registration}}">
                                                        {{element.type_of_registration}}</option>
                                                    {% endfor %}
                                                </select>
                                            </div>
                                        </div>

                                        <div class="form-group col-lg-4">
                                            <div class="form-group" id="company_non_company1211">
                                                <label class="col-form-label">Company Non Company:</label>
                                                <select style="width:160px;" class="custom-select"
                                                    id="company_non_company1" name="company_non_company"
                                                    default="company">
                                                    <option value="NA">NA</option>
                                                    {% for element in company_non_company %}
                                                    <option value="{{element.company_non_company}}"
                                                        {% if element.company_non_company == 'Company' %}selected{% endif %}>
                                                        {{element.company_non_company}}</option>
                                                    {% endfor %}
                                                </select>
                                            </div>
                                        </div>

                                        <div class="form-group col-lg-4">
                                            <div class="form-group" id="capital_or_input1211">
                                                <label class="col-form-label">Capital OR Input</label>
                                                <select style="width:160px;" class="custom-select"
                                                    id="capital_or_input1" name="capital_or_input">
                                                    <!-- <option value="NA">NA</option> -->
                                                    {% for element in capital_input %}
                                                    <option value="{{element.capital_input}}"
                                                        {% if element.capital_input == 'Capital' %}selected{% endif %}>
                                                        {{element.capital_input}}</option>
                                                    {% endfor %}
                                                </select>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="row">
                                        <div class="form-group col-lg-4">
                                            <div class="form-group" id="tds_section1211">
                                                <label class="col-form-label">TDS Section</label>
                                                <select style="width:160px;" class="custom-select" id="tds_section1"
                                                    name="tds_section">
                                                    <option value="NA">NA</option>
                                                    {% for element in tds_section %}
                                                    <option value="{{element.tds_section}}"
                                                        {% if element.tds_section == '194C2' %}selected{% endif %}>
                                                        {{element.tds_section}}</option>
                                                    {% endfor %}
                                                </select>
                                            </div>
                                        </div>
                                        <div class="form-group col-lg-4">
                                            <div class="form-group" id="supplier_customer1211">
                                                <label class="col-form-label">Supplier or Employee</label>
                                                <select style="width:160px;" class="custom-select"
                                                    id="supplier_customer1" name="supplier_or_customer_or_employee">
                                                    <!-- <option value="NA">NA</option> -->
                                                    {% for element in supplier_customer %}
                                                    <option value="{{element.supplier_customer}}"
                                                        {% if element.supplier_customer == 'customer' %}selected{% endif %}>
                                                        {{element.supplier_customer}}</option>
                                                    {% endfor %}
                                                </select>
                                            </div>
                                        </div>
                                        <div class="form-group col-lg-4">
                                            <div class="form-group" id="tds_percentage1211">
                                                <label class="col-form-label">TDS Percentage</label>
                                                <select style="width:160px;" class="custom-select" id="tds_percentage1"
                                                    name="tds_percentage">
                                                    <option value="NA">NA</option>
                                                    {% for element in tds_section %}
                                                    <option value="{{element.tds_percentage}}"
                                                        {% if element.tds_percentage == '18%' %}selected{% endif %}>
                                                        {{element.tds_percentage}}</option>
                                                    {% endfor %}
                                                </select>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="row">
                                        <div class="form-group col-lg-4">
                                            <div class="form-group">
                                                <label class="col-form-label">Type of Service</label>
                                                <select style="width:160px;" class="custom-select" style="width:160px;"
                                                    id="type_of_service1" name="type_of_service">
                                                    <option value="NA">NA</option>
                                                    {% for services in type_of_services %}
                                                    <option value="{{services.type_of_service}}"
                                                        {% if services.type_of_service == 'Import of Services' %}selected{% endif %}>
                                                        {{services.type_of_service}}</option>
                                                    {% endfor %}
                                                </select>
                                            </div>
                                        </div>
                                        <div class="form-group col-lg-4">
                                            <div class="form-group">
                                                <label for="example-text-input" class="col-form-label">Opening
                                                    Balance</label>
                                                <input class="form-control" type="text" name="opening_balance" value=""
                                                    id="opening_balance1">
                                            </div>
                                        </div>

                                        <div class="form-group col-lg-4">
                                            <div class="form-group">
                                                <label for="example-text-input" class="col-form-label">Goods and
                                                    Services</label>
                                                <input class="form-control" type="text" name="goods_or_services"
                                                    value="" id="goods_or_services1">
                                            </div>
                                        </div>


                                        <div class="form-group col-lg-4">
                                            <div class="form-group">
                                                <input class="form-control" type="hidden" name="company_code"
                                                    id="company_code">
                                            </div>
                                        </div>
                                    </div>

                                    <div class="row">
                                        <div class="form-group col-lg-4">
                                            <div class="form-group" id="maintain_balance_bill_by_bill1211">
                                                <label class="col-form-label">Maintain Balance BillByBill <span
                                                        style='color:red'>*</span> </label>
                                                <select style="width:160px;" class="custom-select"
                                                    id="maintain_balance_bill_by_bill1"
                                                    name="maintain_balance_bill_by_bill" required>
                                                    <option value="Yes" selected>Yes</option>
                                                    <option value="No">No</option>
                                                </select>
                                            </div>
                                        </div>
                                        <div class="form-group col-lg-4">
                                            <div class="form-group" id="credit_period1211">
                                                <label for="example-text-input" class="col-form-label">credit
                                                    Period</label>
                                                <input class="form-control" type="text" name="credit_period" value=""
                                                    id="credit_period1">
                                            </div>
                                        </div>

                                        <div class="form-group col-lg-4">
                                            <div class="form-group" id="credit_limit1211">
                                                <label for="example-text-input" class="col-form-label">Credit
                                                    Limit</label>
                                                <input class="form-control" type="text" name="credit_limit" value=""
                                                    id="credit_limit1">
                                            </div>
                                        </div>
                                    </div>

                                    <div class="row">
                                        <div class="form-group col-lg-4">
                                            <div class="form-group" id="rd1211">
                                                <label for="example-text-input" class="col-form-label">RD OR URD</label>
                                                <input class="form-control" type="text" name="rd_or_urd" value="URD"
                                                    id="rd" readonly>
                                            </div>
                                        </div>

                                        <div class="form-group col-lg-4">
                                            <div class="form-group" id="pan_no1211">
                                                <label class="col-form-label">Pan No.</label>
                                                <input type="text" class="form-control" name="pan_no" id="pan_no1"
                                                    value="">
                                            </div>
                                        </div>
                                        <div class="form-group col-lg-4">
                                            <div class="form-group" id="gst_no1211">
                                                <label class="col-form-label">GST No.</label>
                                                <input type="text" class="form-control" name="gst_no" id="gst_no1"
                                                    value="">
                                            </div>
                                        </div>
                                    </div>
                                    <!-- </div> -->
                                    <!-- <div class="container mySlides fade"> -->
                                    <div class="row">
                                        <div class="form-group col-lg-4" id="address1211">
                                            <div class="form-group">
                                                <label for="example-text-input" class="col-form-label">Address:</label>
                                                <input class="form-control" type="text" name="address" value=""
                                                    id="address1">
                                            </div>
                                        </div>
                                        <div class="form-group col-lg-4" id="city1211">
                                            <div class="form-group">
                                                <label for="example-text-input" class="col-form-label">City</label>
                                                <input class="form-control" type="text" name="city" value="" id="city1">
                                            </div>
                                        </div>
                                        <div class="form-group col-lg-4">
                                            <div class="form-group" id="state1211">
                                                <!-- <div class="input-group-prepend"> -->
                                                <label for="example-text-input" class="col-form-label">Place of Supply
                                                    State</label>
                                                <!-- </div> -->
                                                <select style="width:160px;" class="custom-select" id="state1121"
                                                    name="state" required>

                                                    {% for element in place_of_supply_state %}
                                                    <option value="{{element.state}}"
                                                        {% if element.state == 'Haryana' %}selected{% endif %}>
                                                        {{element.state}}</option>
                                                    {% endfor %}
                                                </select>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="row">
                                        <div class="form-group col-lg-4">
                                            <div class="form-group" id="country1211">
                                                <!-- <div class="input-group-prepend"> -->
                                                <label for="example-text-input" class="col-form-label">Place of Supply
                                                    Country</label>
                                                <select style="width:160px;" class="custom-select" id="country11211"
                                                    name="state" required>

                                                    {% for element in place_of_supply_country %}
                                                    <option value="{{element.country_name}}"
                                                        {% if element.country_name == 'India' %}selected{% endif %}>
                                                        {{element.country_name}}</option>
                                                    {% endfor %}
                                                </select>
                                            </div>
                                        </div>
                                        <div class="form-group col-lg-4">
                                            <div class="form-group" id="pin_code1211">
                                                <label for="example-text-input" class="col-form-label">Pin Code</label>
                                                <input class="form-control" type="text" name="pin_code" value=""
                                                    id="pin_code1">
                                            </div>
                                        </div>
                                        <div class="form-group col-lg-4">
                                            <div class="form-group" id="contact_no1211">
                                                <label for="example-text-input" class="col-form-label">Contact
                                                    No.</label>
                                                <input class="form-control" type="number" name="contact_no" value=""
                                                    id="contact_no1">
                                            </div>
                                        </div>
                                    </div>
                                    <div class="row">
                                        <div class="form-group col-lg-4">
                                            <div class="form-group" id="email1211">
                                                <label for="example-email-input" class="col-form-label">Email ID</label>
                                                <input class="form-control" type="email" name="emailid" value=""
                                                    id="emailid1">
                                            </div>
                                        </div>
                                        <div class="form-group col-lg-4">
                                            <div class="form-group" id="bank_ifsc_code2211">
                                                <label for="example-text-input" class="col-form-label">Bank ifsc
                                                    code1:</label>
                                                <input class="form-control" type="text" name="bank_ifsc_code1" value=""
                                                    id="bank_ifsc_code21">
                                            </div>
                                        </div>

                                        <div class="form-group col-lg-4">
                                            <div class="form-group" id="bank_account_holder_name1211">
                                                <label for="example-text-input" class="col-form-label">Bank account
                                                    holder name1:</label>
                                                <input class="form-control" type="text" name="bank_account_holder_name1"
                                                    value="" id="bank_account_holder_name11">
                                            </div>
                                        </div>
                                    </div>
                                    <!-- </div> -->
                                    <!-- <div class="container mySlides fade"> -->
                                    <div class="row">
                                        <div class="form-group col-lg-4">
                                            <div class="form-group" id="bank_name1211">
                                                <label for="example-text-input" class="col-form-label">Bank
                                                    name1</label>
                                                <input class="form-control" type="text" name="bank_name1" value=""
                                                    id="bank_name11">
                                            </div>
                                        </div>
                                        <div class="form-group col-lg-4">
                                            <div class="form-group" id="bank_account_no1211">
                                                <label for="example-email-input" class="col-form-label">Bank account
                                                    no1:</label>
                                                <input class="form-control" type="number" name="bank_account_no1"
                                                    value="" id="bank_account_no11">
                                            </div>
                                        </div>
                                        <div class="form-group col-lg-4">
                                            <div class="form-group" id="bank_ifsc_code1211">
                                                <label for="example-text-input" class="col-form-label">Bank ifsc
                                                    code1:</label>
                                                <input class="form-control" type="text" name="bank_ifsc_code1" value=""
                                                    id="bank_ifsc_code11">
                                            </div>
                                        </div>
                                    </div>
                                    <div class="row">
                                        <div class="form-group col-lg-4">
                                            <div class="form-group" id="bank_account_holder_name2211">
                                                <label for="example-text-input" class="col-form-label">Bank account
                                                    holder name2:</label>
                                                <input class="form-control" type="text" name="bank_account_holder_name2"
                                                    value="" id="bank_account_holder_name21">
                                            </div>
                                        </div>
                                        <div class="form-group col-lg-4">
                                            <div class="form-group" id="bank_name2211">
                                                <label for="example-text-input" class="col-form-label">Bank
                                                    name2</label>
                                                <input class="form-control" type="text" name="bank_name2" value=""
                                                    id="bank_name21">
                                            </div>
                                        </div>
                                        <div class="form-group col-lg-4">
                                            <div class="form-group" id="bank_account_no2211">
                                                <label for="example-email-input" class="col-form-label">Bank account
                                                    no2:</label>
                                                <input class="form-control" type="number" name="bank_account_no2"
                                                    value="" id="bank_account_no21">
                                            </div>
                                        </div>
                                    </div>

                                    <!-- </div> -->
                                    <div class="form-group">
                                        <center><button type="submit"
                                                class="btn btn-primary mt-4 pr-4 pl-4">Submit</button></center>
                                    </div>
                                    <br>
                                </form>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
<script>
        (function () {
            var sel = $('#head1');
            var selected = sel.val(); // cache selected value, before reordering
            var opts_list = sel.find('option');
            opts_list.sort(function (a, b) { return $(a).text() > $(b).text() ? 1 : -1; });
            sel.html('').append(opts_list);
            sel.val(selected);
        });

        $(function () {
            var sel = $('#inward_outward1');
            var selected = sel.val(); // cache selected value, before reordering
            var opts_list = sel.find('option');
            opts_list.sort(function (a, b) { return $(a).text() > $(b).text() ? 1 : -1; });
            sel.html('').append(opts_list);
            sel.val(selected);
        });
        $(function () {
            var sel = $('#company_non_company1');
            var selected = sel.val(); // cache selected value, before reordering
            var opts_list = sel.find('option');
            opts_list.sort(function (a, b) { return $(a).text() > $(b).text() ? 1 : -1; });
            sel.html('').append(opts_list);
            sel.val(selected);
        });
        $(function () {
            var sel = $("#type_of_registration1");
            var selected = sel.val(); // cache selected value, before reordering
            var opts_list = sel.find('option');
            opts_list.sort(function (a, b) { return $(a).text() > $(b).text() ? 1 : -1; });
            sel.html('').append(opts_list);
            sel.val(selected);
        });
        $(function () {
            var sel = $('#type_of_account1');
            var selected = sel.val(); // cache selected value, before reordering
            var opts_list = sel.find('option');
            opts_list.sort(function (a, b) { return $(a).text() > $(b).text() ? 1 : -1; });
            sel.html('').append(opts_list);
            sel.val(selected);
        });
        $(function () {
            var sel = $('#supplier_or_customer_or_employee1');
            var selected = sel.val(); // cache selected value, before reordering
            var opts_list = sel.find('option');
            opts_list.sort(function (a, b) { return $(a).text() > $(b).text() ? 1 : -1; });
            sel.html('').append(opts_list);
            sel.val(selected);
        });
        $(function () {
            var sel = $('#capital_or_input1');
            var selected = sel.val(); // cache selected value, before reordering
            var opts_list = sel.find('option');
            opts_list.sort(function (a, b) { return $(a).text() > $(b).text() ? 1 : -1; });
            sel.html('').append(opts_list);
            sel.val(selected);
        });

        $(function () {
            var sel = $('#tds_section1');
            var selected = sel.val(); // cache selected value, before reordering
            var opts_list = sel.find('option');
            opts_list.sort(function (a, b) { return $(a).text() > $(b).text() ? 1 : -1; });
            sel.html('').append(opts_list);
            sel.val(selected);
        });
        $(function () {
            var sel = $('#tds_percentage1');
            var selected = sel.val(); // cache selected value, before reordering
            var opts_list = sel.find('option');
            opts_list.sort(function (a, b) { return $(a).text() > $(b).text() ? 1 : -1; });
            sel.html('').append(opts_list);
            sel.val(selected);
        });
        $(function () {
            var sel = $('#state1121');
            var selected = sel.val(); // cache selected value, before reordering
            var opts_list = sel.find('option');
            opts_list.sort(function (a, b) { return $(a).text() > $(b).text() ? 1 : -1; });
            sel.html('').append(opts_list);
            sel.val(selected);
        });
        $(function () {
            var sel = $('#country11211');
            var selected = sel.val(); // cache selected value, before reordering
            var opts_list = sel.find('option');
            opts_list.sort(function (a, b) { return $(a).text() > $(b).text() ? 1 : -1; });
            sel.html('').append(opts_list);
            sel.val(selected);
        });



        $(function () {
            var sel = $('#type_of_service1121');
            var selected = sel.val(); // cache selected value, before reordering
            var opts_list = sel.find('option');
            opts_list.sort(function (a, b) { return $(a).text() > $(b).text() ? 1 : -1; });
            sel.html('').append(opts_list);
            sel.val(selected);
        });

        $(document).ready(function () {

            $("#country11211").select2();
            $("#state1121").select2();
            $("#tds_percentage1").select2();
            $("#tds_section1").select2();
            $("#capital_or_input1").select2();
            $("#supplier_customer1").select2();
            $("#type_of_account1").select2();
            $("#type_of_registration1").select2();
            $("#company_non_company1").select2();
            $("#inward_outward1").select2();
            $("#head1").select2();
            $("#type_of_service1").select2();
        });



        $("#type_of_account1").on("change", function () {
            if ($("#type_of_account1").val() == "Nominal") {
                $("#pan_no1211").attr("hidden", true);
                $("#gst_no1211").attr("hidden", true);
                $("#address1211").attr("hidden", true);
                $("#city1211").attr("hidden", true);
                $("#state1211").attr("hidden", true);
                $("#country1211").attr("hidden", true);
                $("#pin_code1211").attr("hidden", true);
                $("#contact_no1211").attr("hidden", true);
                $("#email1211").attr("hidden", true);
                $("#bank_account_holder_name1211").attr("hidden", true);
                $("#bank_name1211").attr("hidden", true);
                $("#bank_account_no1211").attr("hidden", true);
                $("#bank_ifsc_code1211").attr("hidden", true);
                $("#bank_account_holder_name2211").attr("hidden", true);
                $("#bank_name2211").attr("hidden", true);
                $("#bank_account_no2211").attr("hidden", true);
                $("#bank_ifsc_code2211").attr("hidden", true);


                $("#company_non_company1211").attr("hidden", true);
                $("#type_of_registration1211").attr("hidden", true);
                $("#capital_or_input1211").attr("hidden", true);
                $("#tds_section1211").attr("hidden", true);
                $("#supplier_customer1211").attr("hidden", true);
                $("#tds_percentage1211").attr("hidden", true);
                $("#credit_period1211").attr("hidden", true);
                $("#maintain_balance_bill_by_bill1211").attr("hidden", true);
                $("#credit_limit1211").attr("hidden", true);
                $("#rd1211").attr("hidden", true);



            }
            else {
                $("#pan_no1211").attr("hidden", false);
                $("#gst_no1211").attr("hidden", false);
                $("#address1211").attr("hidden", false);
                $("#city1211").attr("hidden", false);
                $("#state1211").attr("hidden", false);
                $("#country1211").attr("hidden", false);
                $("#pin_code1211").attr("hidden", false);
                $("#contact_no1211").attr("hidden", false);
                $("#email1211").attr("hidden", false);
                $("#bank_account_holder_name1211").attr("hidden", false);
                $("#bank_name1211").attr("hidden", false);
                $("#bank_account_no1211").attr("hidden", false);
                $("#bank_ifsc_code1211").attr("hidden", true);
                $("#bank_account_holder_name2211").attr("hidden", true);
                $("#bank_name2211").attr("hidden", true);
                $("#bank_account_no2211").attr("hidden", true);
                $("#bank_ifsc_code2211").attr("hidden", true);


                $("#company_non_company1211").attr("hidden", false);
                $("#type_of_registration1211").attr("hidden", false);
                $("#capital_or_input1211").attr("hidden", false);
                $("#tds_section1211").attr("hidden", false);
                $("#supplier_customer1211").attr("hidden", false);
                $("#tds_percentage1211").attr("hidden", false);
                $("#credit_period1211").attr("hidden", false);
                $("#maintain_balance_bill_by_bill1211").attr("hidden", false);
                $("#credit_limit1211").attr("hidden", false);
                $("#rd1211").attr("hidden", false);

            }

        });

        $("#gst_no1").on("change", function () {
            var gstno = $("#gst_no1").val();
            if (gstno == "") {
                $("#rd").val("URD")
            }
            else {
                $("#rd").val("RD")
            }
        }
        );
        $("#gst_no").on("change", function () {
            var gstno = $("#gst_no").val();
            if (gstno == "") {
                $("#urd").val("URD")
            }
            else {
                $("#urd").val("RD")
            }
        }
        );

        $(document).ready(function () {
            var gstno = $("#gst_no").val();
            if (gstno == "") {
                $("#urd").val("URD")
            }
            else {
                $("#urd").val("RD")
            }
        });

        $(document).ready(function () {
            var gstno = $("#gst_no1").val();
            if (gstno == "") {
                $("#rd").val("URD")
            }
            else {
                $("#rd").val("RD")
            }
        });

        $("#ledger_name1").change(function () {
            $("#vendor_name1").val($(this).val());
        });

        $("#ledger_name").change(function () {
            $("#vendor_name").val($(this).val());
        });
    </script>
</body>
</html>

